import pandas as pd
import os
import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def load_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File CSV '{csv_path}' tidak ditemukan!")
    
    df = pd.read_csv(csv_path)
    required_cols = {"Nama", "Acuan Manual", "Split"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"File CSV harus memiliki kolom: {required_cols}")
    
    df["Split"] = df["Split"].astype(str).str.lower().str.strip()
    
    tuning_dict = df[df["Split"] == "tuning"].set_index("Nama")["Acuan Manual"].to_dict()
    test_dict = df[df["Split"] == "test"].set_index("Nama")["Acuan Manual"].to_dict()
    
    return tuning_dict, test_dict

_PREP_CACHE = {}

class RedBloodCellPipeline:
    def __init__(self, peak_ratio=0.45, min_area=2500, max_hole_area=4000, min_solidity=0.75, border_area_factor=0.4, 
                 min_wbc_area=600, seed_min_dist_ratio= 0.5, wbc_margin=20, close_kernel=9, dist_sigma=1, cell_radius=40, 
                 bg_diff_thresh=10, bg_kernel=141, hole_convexity=0.6, min_lab_a=0, min_lab_b=0, hole_circularity=0.0,
                   silver_kernel=9, area_split_ratio=0.0, seed_fallback=True):
        # self.kernel_size = kernel_size
        # self.peak_ratio = peak_ratio
        # self.min_area = min_area
        self.peak_ratio = peak_ratio
        self.min_area = min_area #luas sel utuh
        self.max_hole_area = max_hole_area #luas maks lubang yang diisi
        self.min_solidity = min_solidity #buang fegmen bergerigi
        self.border_area_factor = border_area_factor #sel terpotong terpi citra boleh lebih kecil
        self.min_wbc_area = min_wbc_area #
        self.seed_min_dist_ratio = seed_min_dist_ratio #jarak minimal antar seed relatif thd jari jari
        self.wbc_margin = wbc_margin #margin sitoplasma di sekitar inti WBC
        self.close_kernel = close_kernel #menutu celah kecil pada cicin sel sebelum filling
        self.dist_sigma = dist_sigma
        self.cell_radius = cell_radius
        self.bg_diff_thresh = bg_diff_thresh #selisih gelap minimum thd latar lokal
        self.bg_kernel = bg_kernel #harus lebih dari diameter sel
        self.hole_convexity = hole_convexity #min. rasio luas/hull lubang diisi
        self.min_lab_a = min_lab_a
        self.min_lab_b = min_lab_b
        self.hole_circularity = hole_circularity #kebulatan lubang; celah antar sel cekung 
        self.silver_kernel = silver_kernel
        self.area_split_ratio = area_split_ratio
        self.seed_fallback = seed_fallback

    def fill_cell_holes(self, binary_mask):
        binary_mask = np.ascontiguousarray(binary_mask, dtype=np.uint8)
        h, w = binary_mask.shape
        filled = binary_mask.copy()
        inv_mask = np.ascontiguousarray(cv2.bitwise_not(binary_mask), dtype=np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(inv_mask, connectivity=4)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
                continue
            if area > self.max_hole_area:
                continue
            comp = np.ascontiguousarray(lab[y:y + bh, x:x + bw] == i, dtype=np.uint8)
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            cnt = max(cs, key=cv2.contourArea)
            cnt_area = cv2.contourArea(cnt)
            if cnt_area > self.max_hole_area:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * cnt_area / (perimeter * perimeter) if perimeter > 0 else 0
            convexity = cnt_area / hull_area if hull_area > 0 else 0
            if convexity >= self.hole_convexity and circularity >= self.hole_circularity:
                    cv2.drawContours(filled, [cnt], -1, 255, -1, offset=(int(x), int(y)))
        return filled

    def _detect_wbc(self, img):
        lab_chan = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 2]
        sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
        _, nuc = cv2.threshold(sat, 100, 255, cv2.THRESH_BINARY)
        nuc = cv2.morphologyEx(nuc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        nuc_closed = cv2.morphologyEx(nuc, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(nuc_closed)
        wbc = np.zeros_like(nuc)
        k_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.wbc_margin + 1, 2 * self.wbc_margin + 1))

        for i in range ( 1, n):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_wbc_area:
                comp = np.uint8(lab == i) * 255
                search_zone = cv2.dilate(comp, k_margin) #perluas area pencarian disekitar inti ini untuk menangkap sitoplasman WBC
                cyto = ((sat > 45) | (lab_chan < 126)).astype(np.uint8) * 255
                wbc_local = cv2.bitwise_and(search_zone, cyto)
                wbc_local = cv2.bitwise_or(wbc_local, comp)

                contours, _ = cv2.findContours(wbc_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    all_pts = np.vstack(contours)
                    hull = cv2.convexHull(all_pts)
                    cv2.fillConvexPoly(wbc, hull, 255)

        if cv2.countNonZero(wbc) > 0:
            k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            wbc = cv2.dilate(wbc, k_smooth)

        return wbc
    def _region_oke(self, mask_sel, c, area, lab_img, h, w):
        x, y, bw, bh = cv2.boundingRect(c)
        on_border = x <= 0 or y <= 0 or x + bw >= w or y + bh >= h
        limit = self.min_area * self.border_area_factor if on_border else self.min_area
        if area < limit:
            return False
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = cv2.contourArea(c) / hull_area if hull_area > 0 else 0
        if not on_border and solidity < self.min_solidity:
            return False
        lm = lab_img[mask_sel > 0].mean(axis=0)
        if lm[1] < self.min_lab_a or lm[2] < self.min_lab_b:
            return False
        return True

    def _prepare(self, img_path):
        key = (img_path, self.wbc_margin, self.min_wbc_area, self.bg_kernel)
        if key not in _PREP_CACHE:
            img = cv2.imread(img_path)
            if img is None:
                return None
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            #perataan nois dan pemrosesan citra awal
            wbc_mask = self._detect_wbc(img)
            g = cv2.GaussianBlur(img[:, :, 1], (7, 7), 0)
            g_nw = g.copy()
            g_nw[wbc_mask > 0] = int(np.percentile(g, 90))      # WBC jangan ikut menggelapkan estimasi latar
            bg = cv2.morphologyEx(g_nw, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.bg_kernel, self.bg_kernel)))
            bg = cv2.GaussianBlur(bg, (0, 0), 25)
            diff = np.clip(bg.astype(np.int16) - g.astype(np.int16), 0, 255).astype(np.uint8)
            diff[wbc_mask > 0] = 0
            _PREP_CACHE[key] = (img, lab_img, wbc_mask, diff)
        return _PREP_CACHE[key]
    
    def _pipeline_image_processing(self, img_path, save_visual_dir=None):
        prep = self._prepare(img_path)
        if prep is None:
            return 0, 0, None
        img, lab_img, wbc_mask, diff = prep
        h, w = img.shape[:2]
        
        _, thresh = cv2.threshold(diff, self.bg_diff_thresh, 255, cv2.THRESH_BINARY)

        #bersihkan noise titik pada citra
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        if self.close_kernel > 1:
            opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel)))
        clean_mask = self.fill_cell_holes(opening)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        clean_mask[wbc_mask > 0] = 0

        #pipeline baseline (tanpa pemisah objek bersentuhan)
        #memakai filter region yang sama dengan watershed (pada fungsi _region_ok)
        n_base, base_labels = cv2.connectedComponents(clean_mask)
        baseline_sel = 0
        min_keep = min(self.min_area, self.min_wbc_area * self.border_area_factor)
        base_areas = np.bincount(base_labels.ravel(), minlength=n_base)
        #baseline_sel = sum(1 for i in range(1, n_base) if stats_base[i, cv2.CC_STAT_AREA] >= self.min_area)
        for i in range(1, n_base):
            if base_areas[i] < min_keep:
                continue
            mask_b = np.ascontiguousarray((base_labels == i).astype(np.uint8) * 255)
            cnts_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts_b:
                continue
            cb = max(cnts_b, key=cv2.contourArea)
            if self._region_oke(mask_b, cb, cv2.countNonZero(mask_b), lab_img, h, w):
                baseline_sel += 1

        #pipeline perbaikan (transformasi jarak dan watersheed)
        #sure_bg = cv2.dilate(opening, kernel, iterations=3)

        #transformasi jarak pada clean mask
        dist = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
        dist = cv2.GaussianBlur(dist, (0, 0), self.dist_sigma)
        maxd = float(dist.max())
        if dist.max() == 0:
            return baseline_sel, 0, img.copy()

        #seed puncak lokal peta jarak
        R = float(self.cell_radius) if self.cell_radius else maxd
        coords = peak_local_max(
            dist,
            min_distance=max(3, int(self.seed_min_dist_ratio * R)),
            threshold_abs=self.peak_ratio * R,
            labels=(clean_mask > 0).astype(np.uint8),
            exclude_border=False,
        )

        coords = np.asarray(coords).reshape(-1, 2)
        if self.seed_fallback:
            n_cc, cc_labels = cv2.connectedComponents(clean_mask)
            seeded = {int(cc_labels[r, c]) for r, c in coords}
            extra = []
            for i in range(1, n_cc):
                if i in seeded:
                    continue
                comp = cc_labels == i
                if comp.sum() < self.min_area * self.border_area_factor:
                    continue
                extra.append(np.unravel_index(int(np.argmax(np.where(comp, dist, -1.0))), dist.shape))
            if extra:
                coords = np.vstack([coords, np.array(extra).reshape(-1, 2)])
 
        seeds = np.zeros(dist.shape, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            seeds[r, c] = i
        markers = watershed(-dist, seeds, mask=(clean_mask > 0)) if len(coords) else np.zeros(dist.shape, np.int32)
        #hitung objek dan gambar penanda geometri pada citra
        img_penanda = img.copy()
        watershed_sel = 0
    
        accepted = []
        ws_areas = np.bincount(markers[clean_mask > 0].ravel(), minlength=int(markers.max()) + 1)
        for label in range(1, int(markers.max()) + 1):
            if ws_areas[label] < min_keep:
                continue
            mask_sel = (((markers == label) & (clean_mask > 0)).astype(np.uint8)) * 255
            mask_sel = np.ascontiguousarray(mask_sel, dtype=np.uint8)
            area = cv2.countNonZero(mask_sel)
            contours, _ = cv2.findContours(mask_sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            if not self._region_oke(mask_sel, c, area, lab_img, h, w):
                continue
            bx, by, bw_, bh_ = cv2.boundingRect(c)
            on_border = bx <= 0 or by <= 0 or bx + bw_ >= w or by + bh_ >= h
            accepted.append((c, area, on_border))


        weights = [1] * len(accepted)
        if self.area_split_ratio > 0 and accepted:
            interior = [a_ for _, a_, ob in accepted if not ob]
            ref = float(np.median(interior if len(interior) >= 3 else [a_ for _, a_, _ in accepted]))
            weights = [max(1, int(round(a_ / ref))) if a_ > self.area_split_ratio * ref else 1 for _, a_, _ in accepted]


        for (c, area, _), wt in zip(accepted, weights):
            watershed_sel += wt
            cv2.drawContours(img_penanda, [c], -1, (0, 255, 0), 2)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.circle(img_penanda, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(img_penanda, str(watershed_sel), (cx - 8, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        #simpan hasil visualisasi objek
        if save_visual_dir:
            os.makedirs(save_visual_dir, exist_ok=True)
            base_name = os.path.basename(img_path)
            cv2.rectangle(img_penanda, (0, 0), (400, 32), (0, 0, 0), -1)
            cv2.putText(img_penanda, f"Baseline: {baseline_sel} Watershed_sel: {watershed_sel}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(save_visual_dir, f"penanda_{base_name}"), img_penanda)
            cv2.imwrite(os.path.join(save_visual_dir, f"mask_utuh_{base_name}"), clean_mask)
            dbg = img.copy()
            for r, c, in coords:
                cv2.circle(dbg, (int(c), int(r)), 5, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(save_visual_dir, f"seed_{base_name}"), dbg)
        return baseline_sel, watershed_sel, img_penanda
        
    def process_inference_batch(self, img_dir, manual_git_dict, save_visual_dir="output_visual"):
        records = []
        
        for file_name, acuan_manual in manual_git_dict.items():
            img_path = os.path.join(img_dir, f"{file_name}.jpg")
            # Eksekusi pipeline pemrosesan citra
            hasil_base, hasil_ws, _ = self._pipeline_image_processing(img_path, save_visual_dir)
            
            records.append({
                "File Citra": file_name,
                "acuan_manual": acuan_manual,
                "Prediksi Baseline (Tanpa Pemisahan)": hasil_base,
                "Prediksi Watershed (Dengan Pemisahan)": hasil_ws,
                "error base_line": hasil_base - acuan_manual,
                "error watershed": hasil_ws - acuan_manual
            })
            
        df = pd.DataFrame(records)
        return df

if __name__ == "__main__":
    BASE = 
    TEST_IMG_FOLDER = os.path.join(BASE, "Testing", "Images")
    CSV_PATH = 
    _, TEST_MANUAL = load_csv(CSV_PATH)

    BEST = {                       
        "peak_ratio": 0.45,
        "seed_min_dist_ratio": 0.6,
        "bg_diff_thresh": 11,
        "close_kernel":5,
        "cell_radius": 40,
        "max_hole_area":2500,
        "hole_convexity":0.0,
        "hole_circularity": 0.0,
        "wbc_margin":28,
        "border_area_factor":0.4,
        "min_solidity": 0.6,        # Dilonggarkan dari 0.75 agar potongan watershed tidak banyak yang terbuang
        "dist_sigma": 0.5,
        "min_lab_a": 0,
        "min_lab_b": 0,
        "silver_kernel":9,
        "area_split_ratio":0.0
    }
    pipeline = RedBloodCellPipeline(**BEST)

    # CITRA_UJI_BARU = { 
    #     "BloodImage_00398":24, 
    #     "BloodImage_00393":20, 
    #     "BloodImage_00396":27,
    #     "BloodImage_00352":20, 
    #     "BloodImage_00354":26,
    # }

    df_hasil = pipeline.process_inference_batch(TEST_IMG_FOLDER, TEST_MANUAL, save_visual_dir="output_visual")
    # df_hasil["Galat Baseline"] = df_hasil["Prediksi Baseline (Tanpa Pemisahan)"] - df_hasil["Acuan (XML)"]
    # df_hasil["Galat Watershed"] = df_hasil["Prediksi Watershed (Dengan Pemisahan)"] - df_hasil["Acuan (XML)"]

    print("\nHASIL HITUNG SEL DARAH MERAH")
    print(df_hasil.to_string(index=False))
    print(f"\nMAE baseline : {df_hasil['error base_line'].abs().mean():.2f}")
    print(f"MAE watershed: {df_hasil['error watershed'].abs().mean():.2f}")
    print(f"Bias watershed: {df_hasil['error watershed'].mean():+.2f}")
    print(f"Bias baseline : {df_hasil['error base_line'].mean():+.2f}")

    df_hasil.to_csv("hasil_pengujian_utama.csv", index=False)
    print("-> Visual di 'output_visual/', tabel di 'hasil_pengujian_utama.csv'")