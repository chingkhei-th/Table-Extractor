# pdf_processor.py (alternative version using PyMuPDF)
import fitz  # PyMuPDF
from PIL import Image
import io

def convert_pdf_to_images(pdf_path, dpi=300):
    """
    Convert a PDF file to a list of PIL Images using PyMuPDF

    Args:
        pdf_path (str): Path to the PDF file
        dpi (int): DPI for the converted images

    Returns:
        list: List of PIL Images, one per page
    """
    try:
        # Calculate the scaling factor based on DPI
        zoom = dpi / 72  # Default DPI in PDF is 72

        # Open the PDF file
        pdf_document = fitz.open(pdf_path)
        images = []

        # Convert each page to an image
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)

            # Convert page to a matrix with the specified zoom
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert pixmap to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

        pdf_document.close()
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        return []