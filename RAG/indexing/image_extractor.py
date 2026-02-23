from pathlib import Path

import fitz


def extract_images(path: str, out_dir: str = "data/images"):
    images = []
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_stem = Path(path).stem

    with fitz.open(path) as doc:
        for page_idx, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_img = doc.extract_image(xref)
                image_bytes = base_img["image"]
                image_ext = base_img.get("ext", "png")

                file_path = output_dir / f"{pdf_stem}_page{page_idx}_img{img_index}.{image_ext}"
                file_path.write_bytes(image_bytes)

                images.append(
                    {
                        "page": page_idx,
                        "image_path": str(file_path),
                        "source": path,
                    }
                )

    return images
