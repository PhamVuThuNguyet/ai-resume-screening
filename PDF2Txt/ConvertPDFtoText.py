from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import os


def get_all_pdf_paths(directory):
    """
    get all of pdf file path in directory
    :param directory: directory to get file
    :type directory: string
    :return: list of file paths
    :rtype: list
    """
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                file_paths.append(os.path.join(root, file))
    return file_paths


def convert_pdf_to_txt(path_list, save_dir):
    """
    extract text of a pdf file and save to txt file
    :param path_list: list of paths of pdf files to convert
    :type path_list: list of string
    :param save_dir: path to save txt files
    :type save_dir: string
    :return: None
    :rtype:
    """
    for p in path_list:
        path, filename = os.path.split(p)

        resource_manager = PDFResourceManager()
        return_str = StringIO()
        layout_params = LAParams()
        device = TextConverter(resource_manager, return_str, laparams=layout_params)
        interpreter = PDFPageInterpreter(resource_manager, device)

        fp = open(p, 'rb')

        password = ""
        max_pages = 0
        caching = True
        page_nos = set()
        for page in PDFPage.get_pages(fp, page_nos, maxpages=max_pages, password=password, caching=caching,
                                      check_extractable=True):
            interpreter.process_page(page)

        text = return_str.getvalue()

        file_to_save = save_dir + filename.split(".")[0] + ".txt"
        f = open(file_to_save, "w", encoding="utf-8")
        f.write(text)
        f.close()
        fp.close()
        device.close()
        return_str.close()
        print("Convert", filename, "successfully")


if __name__ == '__main__':
    dir_ = "../data/JD_pdf"
    path_list_ = get_all_pdf_paths(dir_)
    save_dir_ = "../data/converted_txt_JD/"
    convert_pdf_to_txt(path_list_, save_dir_)
