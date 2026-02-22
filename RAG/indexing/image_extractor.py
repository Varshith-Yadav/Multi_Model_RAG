import fitz
import os


def extract_images(path, out_dir="data/images"):
    doc = fitz.open(path)
    images = []
    os.makedirs(out_dir, exist_ok=True)

    for page_idx, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_img = doc.extract_image(xref)
            image_bytes = base_img["image"]

            file_path = f"{out_dir}/page{page_idx}_img{img_index}.png"

            with open(file_path, "wb") as f:
                f.write(image_bytes)

            images.append({
                "page": page_idx,
                "image_path": file_path,
                'source': path
            }) 
               
    return images
            