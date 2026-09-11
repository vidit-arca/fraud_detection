import io
import cv2
import numpy as np
from PIL import Image
from scipy.fftpack import dct

class SpatialRichModel:
    """
    Implements standard SRM (Spatial Rich Model) high-pass linear filter kernels
    used in digital image forensics to isolate noise residuals and suppress content.
    """
    # 3 standard 5x5 SRM high-pass filter kernels
    SRM_KERNELS = [
        # 1st order edge/noise residual
        np.array([
            [0,  0,  0,  0,  0],
            [0, -1,  2, -1,  0],
            [0,  2, -4,  2,  0],
            [0, -1,  2, -1,  0],
            [0,  0,  0,  0,  0]
        ], dtype=np.float32) / 4.0,
        
        # 2nd order square residual
        np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0,
        
        # 3rd order edge residual
        np.array([
            [0,  0,  1,  0,  0],
            [0,  1, -2,  1,  0],
            [1, -2,  0, -2,  1],
            [0,  1, -2,  1,  0],
            [0,  0,  1,  0,  0]
        ], dtype=np.float32) / 2.0
    ]

    @classmethod
    def extract_srm_residuals(cls, gray_image_np):
        """
        Applies SRM kernels to a single-channel grayscale image (0-255 uint8/float).
        Returns a 3-channel noise residual feature array (H, W, 3).
        """
        img = gray_image_np.astype(np.float32)
        residuals = []
        for kernel in cls.SRM_KERNELS:
            res = cv2.filter2D(img, -1, kernel)
            # Non-linear truncation to focus on micro-residuals (-2.0 to 2.0)
            res = np.clip(res, -3.0, 3.0)
            residuals.append(res)
        return np.stack(residuals, axis=-1)


class SignalForensicsEngine:
    """
    Forensic signal analysis engine executing:
    - Multi-Quality Error Level Analysis (ELA)
    - SRM noise residual map
    - DCT double JPEG compression grid analysis
    - Patch-level noise consistency evaluation
    """

    @staticmethod
    def compute_ela_map(pil_image, quality=90, scale=15.0):
        """
        Computes Error Level Analysis (ELA) by re-compressing at specified quality.
        Returns amplified difference image and mean error score.
        """
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf)

        orig_np = np.array(pil_image.convert("RGB"), dtype=np.float32)
        resaved_np = np.array(resaved.convert("RGB"), dtype=np.float32)

        # Compute amplified delta
        diff = np.abs(orig_np - resaved_np) * scale
        ela_map = np.clip(diff, 0, 255).astype(np.uint8)
        mean_error = float(np.mean(diff))
        
        return ela_map, mean_error

    @staticmethod
    def compute_multiscale_ela(pil_image, qualities=(70, 80, 90, 95)):
        """
        Computes ELA features across multiple JPEG compression qualities.
        """
        results = {}
        for q in qualities:
            ela_map, mean_err = SignalForensicsEngine.compute_ela_map(pil_image, quality=q)
            results[f"ela_q{q}_map"] = ela_map
            results[f"ela_q{q}_mean"] = mean_err
            results[f"ela_q{q}_max"] = float(np.max(ela_map))
            results[f"ela_q{q}_std"] = float(np.std(ela_map))
        return results

    @staticmethod
    def compute_dct_double_compression_score(pil_image):
        """
        Analyzes 8x8 DCT block coefficient distributions to detect double JPEG compression
        artifacts and grid alignment shifts.
        """
        gray = np.array(pil_image.convert("L"), dtype=np.float32)
        h, w = gray.shape
        # Pad to multiple of 8
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="reflect")
            
        h_pad, w_pad = gray.shape
        n_blocks_y = h_pad // 8
        n_blocks_x = w_pad // 8
        
        # Compute 2D DCT for all 8x8 blocks
        ac_coefficients = []
        for i in range(min(n_blocks_y, 60)):  # sample up to 60 rows
            for j in range(min(n_blocks_x, 60)):
                block = gray[i*8:(i+1)*8, j*8:(j+1)*8] - 128.0
                dct_block = dct(dct(block.T, norm="ortho").T, norm="ortho")
                # Sample primary low-frequency AC coefficient (1, 1) and (1, 2)
                ac_coefficients.append(dct_block[1, 1])
                ac_coefficients.append(dct_block[1, 2])
                
        ac_arr = np.array(ac_coefficients)
        # Measure histogram periodicity (peaks caused by double quantization)
        hist, bin_edges = np.histogram(ac_arr, bins=100, range=(-50, 50))
        hist_diff = np.diff(hist)
        zero_crossings = np.sum(np.diff(np.sign(hist_diff)) != 0)
        
        # High zero crossings indicate periodic comb-like pattern of double compression
        periodicity_score = min(1.0, float(zero_crossings) / 45.0)
        return periodicity_score

    @staticmethod
    def analyze_document_signals(pil_image):
        """
        Comprehensive signal forensic analysis of a document.
        Returns composite forensic score (0.0 to 1.0) and detailed diagnostic metrics.
        """
        gray_np = np.array(pil_image.convert("L"))
        
        # 1. SRM noise residuals
        srm_residuals = SpatialRichModel.extract_srm_residuals(gray_np)
        srm_variance = float(np.var(srm_residuals))
        
        # 2. Multi-quality ELA
        ela_data = SignalForensicsEngine.compute_multiscale_ela(pil_image, qualities=(70, 80, 90, 95))
        ela_90_std = ela_data["ela_q90_std"]
        ela_90_max = ela_data["ela_q90_max"]
        
        # 3. DCT Double Compression
        dct_score = SignalForensicsEngine.compute_dct_double_compression_score(pil_image)
        
        # 4. Patch consistency (5x5 grid)
        h, w = gray_np.shape
        patch_h, patch_w = h // 5, w // 5
        patch_noises = []
        for i in range(5):
            for j in range(5):
                patch = gray_np[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                lap = cv2.Laplacian(patch, cv2.CV_64F)
                patch_noises.append(np.std(lap))
                
        noise_disparity = float(np.std(patch_noises) / (np.mean(patch_noises) + 1e-6))
        
        # Composite Signal Score (0.0 = completely authentic, 1.0 = highly manipulated)
        signal_score = float(np.clip(
            0.35 * (noise_disparity / 0.6) + 
            0.35 * (ela_90_std / 18.0) + 
            0.30 * dct_score,
            0.0, 1.0
        ))
        
        return {
            "signal_tamper_score": signal_score,
            "srm_variance": srm_variance,
            "noise_disparity": noise_disparity,
            "dct_double_compression_score": dct_score,
            "ela_metrics": {
                "q90_std": ela_90_std,
                "q90_max": ela_90_max,
                "q80_std": ela_data["ela_q80_std"]
            },
            "srm_residual_map": srm_residuals,
            "ela_heatmap": ela_data["ela_q90_map"]
        }
