import img2pdf
import docx2pdf
import os
import glob
from PIL import Image
import shutil
from subprocess import Popen
import sys


def convert_docx2pdf_libre_office(input_docx, out_folder):
    p = Popen(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir',
               out_folder, input_docx])
    # print(['libreoffice', '--convert-to', 'pdf', input_docx])
    p.communicate()


def convert_img2pdf(img_path, store_path):
    """
    This method used to convert images to pdf bytes
    :param store_path:
    :param img_path: path of image file
    """
    image = Image.open(img_path)
    # Ignore file if size too big
    if image.size[0] > 7000 or image.size[1] > 7000:
        print("This image is too big to convert!")
    else:
        # Remove transparency in the image
        if image.mode in ('RGBA', 'LA', 'PA') or (image.mode == 'P' and 'transparency' in image.info):
            img_path_ = img_path.split(".")[0] + ".jpg"
            # print(img_path_)
            image.load()  # required for png.split()
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
            background.save(img_path_, 'JPEG', quality=80)
            image = Image.open(img_path_)
        # converting into chunks using img2pdf
        pdf_bytes = img2pdf.convert(image.filename)
        # opening or creating pdf file
        file = open(store_path, "wb")
        # writing pdf files with chunks
        file.write(pdf_bytes)


def convert_docx2pdf(docx_path, store_path):
    """
    This method used to convert docx to pdf bytes
    :param docx_path: path to docx file
    :param store_path: Where to store the pdf file
    :return: pdf bytes
    """
    if sys.platform == "darwin" or sys.platform == "win32":
        docx2pdf.convert(docx_path, store_path)
    else:
        # print("Linux detected! Using Libre_office!")
        convert_docx2pdf_libre_office(docx_path, store_path)


def convert_to_pdf(data_path, pdf_output_dir: str = "./pdf/"):
    """

    :param pdf_output_dir: Path where to store the pdf file
    :param data_path: Path where to store data
    :return:
    """
    img_files = list(glob.glob(data_path + "**/*.jpg", recursive=True))
    img_files.extend(glob.glob(data_path + "**/*.png", recursive=True))
    docx_files = glob.glob(data_path + "**/*.docx", recursive=True)
    docx_files.extend(list(glob.glob(data_path + "**/*.doc", recursive=True)))
    pdf_files = glob.glob(data_path + "**/*.pdf", recursive=True)
    os.makedirs(pdf_output_dir, exist_ok=True)

    # Convert img2pdf
    for img_file in img_files:
        file_name = img_file.split("/")[-1]
        image_name = file_name.split(".")[0]
        parent_path = img_file.replace(data_path, '')
        parent_path = parent_path.replace(file_name, '')
        parent_path = pdf_output_dir + "/" + parent_path
        os.makedirs(parent_path, exist_ok=True)
        print("Create path :", parent_path)
        convert_img2pdf(img_file, parent_path + image_name + ".pdf")
        print("Convert file : ", file_name, " to pdf!")
        print("Save pdf file in directory: ", pdf_output_dir, image_name, ".pdf")

    # Convert docx2pdf
    for docx_file in docx_files:
        file_name = docx_file.split("/")[-1]
        docx_name = file_name.split(".")[0]
        parent_path = docx_file.replace(data_path, '')
        parent_path = parent_path.replace(file_name, '')
        parent_path = pdf_output_dir + "/" + parent_path
        os.makedirs(parent_path, exist_ok=True)
        print("Create path :", parent_path)
        convert_docx2pdf(docx_file, parent_path)
        print("Convert file : ", docx_file, " to pdf!")
        print("Save pdf file in directory: ", parent_path, docx_name, ".pdf")

    # Move pdf to pdf folder
    for pdf_file in pdf_files:
        file_name = pdf_file.split("/")[-1]
        pdf_name = file_name.split(".")[0]
        parent_path = pdf_file.replace(data_path, '')
        parent_path = parent_path.replace(file_name, '')
        parent_path = pdf_output_dir + "/" + parent_path
        os.makedirs(parent_path, exist_ok=True)
        print("Create path :", parent_path)
        try:
            shutil.copy(pdf_file, parent_path)
        # If source and destination are same
        except shutil.SameFileError:
            print("Source and destination represents the same file.")
        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")
        print("Convert file :", pdf_file, "to pdf!")
        print("Save pdf file in directory: ", pdf_output_dir + "/" + pdf_name, ".pdf")

    print("Done convert all documents to pdf type. Total ", len(img_files)
          + len(docx_files) + len(pdf_files), " files!")
