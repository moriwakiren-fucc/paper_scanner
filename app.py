import cv2
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image

# ================================
# Streamlit アプリ設定
# ================================
st.set_page_config(page_title="書類スキャナ風アプリ", layout="centered")
st.title("📄 書類スキャナ風アプリ")
st.write("普通のカメラで撮った書類を、スキャナで撮ったように変換します。")

# ================================
# Step 1: 画像アップロード
# ================================
uploaded_file = st.file_uploader("📷 書類写真をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # PILで画像を開いてOpenCV形式に変換
    input_image = Image.open(uploaded_file)
    img = np.array(input_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB→BGR

    st.image(input_image, caption="アップロードした画像", use_column_width=True)

    # ================================
    # Step 2: 書類の輪郭を検出
    # ================================
    st.subheader("🔍 書類領域を検出しています...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # グレースケール化
    blur = cv2.GaussianBlur(gray, (5, 5), 0)      # ノイズ除去
    edged = cv2.Canny(blur, 75, 200)              # エッジ検出

    # 輪郭を検出（外枠を優先して取得）
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_contour = approx
            break

    if doc_contour is not None:
        # ================================
        # Step 3: 台形補正（透視変換）
        # ================================
        st.subheader("📐 台形補正を実行中...")

        # 頂点を整列する関数
        def reorder_points(pts):
            pts = pts.reshape((4, 2))
            rect = np.zeros((4, 2), dtype="float32")

            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)

            rect[0] = pts[np.argmin(s)]      # 左上
            rect[2] = pts[np.argmax(s)]      # 右下
            rect[1] = pts[np.argmin(diff)]   # 右上
            rect[3] = pts[np.argmax(diff)]   # 左下
            return rect

        rect = reorder_points(doc_contour)
        (tl, tr, br, bl) = rect

        # 幅と高さを計算
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        max_width = int(max(width_top, width_bottom))
        max_height = int(max(height_left, height_right))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        # 透視変換行列を計算
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (max_width, max_height))

        # ================================
        # Step 4: 画像の補正・スキャン風加工
        # ================================
        st.subheader("✨ スキャナ風に加工中...")

        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # 適応的閾値処理で白黒を強調
        scanned = cv2.adaptiveThreshold(
            warped_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 10
        )

        # ================================
        # 出力表示
        # ================================
        st.image(scanned, caption="スキャン風に加工された画像", use_column_width=True, clamp=True)

        # ダウンロードボタン
        result = Image.fromarray(scanned)
        buf = BytesIO()
        result.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="📥 加工画像をダウンロード",
            data=byte_im,
            file_name="scanned_document.png",
            mime="image/png"
        )

    else:
        st.warning("書類の輪郭が見つかりませんでした。背景が明るい画像をお試しください。")

else:
    st.info("左のボタンから書類写真をアップロードしてください。")
