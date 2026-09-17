"""
文档解析模块：支持PDF、Word(docx)、Excel(xlsx/xls)
"""
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> str:
    """解析PDF文档，提取纯文本"""
    import fitz  # PyMuPDF
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    text_parts = []
    doc = fitz.open(file_path)
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if page_text.strip():
            text_parts.append(page_text.strip())
    doc.close()
    
    full_text = "\n".join(text_parts)
    if not full_text.strip():
        raise ValueError("PDF文件内容为空或无法提取文本（可能是扫描件）")
    
    logger.info(f"PDF解析完成: {file_path}, 共{len(text_parts)}页, {len(full_text)}字符")
    return full_text


def parse_docx(file_path: str) -> str:
    """解析Word文档(docx)，提取正文段落"""
    from docx import Document
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    doc = Document(file_path)
    paragraphs = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        # 过滤空白段落、纯格式标记
        if text and len(text) > 0:
            paragraphs.append(text)
    
    # 同时提取表格内容
    for table in doc.tables:
        for row in table.rows:
            row_cells = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    row_cells.append(cell_text)
            if row_cells:
                paragraphs.append(" | ".join(row_cells))
    
    full_text = "\n".join(paragraphs)
    if not full_text.strip():
        raise ValueError("Word文档内容为空")
    
    logger.info(f"Word解析完成: {file_path}, 共{len(paragraphs)}段落, {len(full_text)}字符")
    return full_text


def parse_excel(file_path: str) -> str:
    """解析Excel文档，按工作表逐行逐单元格提取"""
    from openpyxl import load_workbook
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 1) 检查是否是合法 xlsx（zip 容器）
    import zipfile
    if not zipfile.is_zipfile(file_path):
        raise ValueError(
            "Excel 文件不是合法的 xlsx 格式（可能后缀名被误改，或文件被损坏）。"
            "请用 Excel/WPS 重新另存为 .xlsx 后再上传。"
        )
    
    # 2) 优先 openpyxl 解析，区分错误类型
    try:
        wb = load_workbook(file_path, data_only=True, read_only=True)
    except Exception as e:
        err = str(e)
        if 'no valid workbook' in err.lower() or 'zip' in err.lower():
            raise ValueError(
                f"Excel 文件结构不完整（可能是下载中断或转换工具输出残缺）。\n"
                f"原始错误: {err}\n"
                f"建议：重新从源文件导出 xlsx，或换一个来源下载。"
            ) from e
        if 'password' in err.lower() or 'encrypted' in err.lower():
            raise ValueError("Excel 文件已加密，请先解密后再上传。") from e
        raise  # 其他错误原样抛出
    
    sheets_text = []
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        sheet_rows = []
        
        for row in ws.iter_rows(values_only=True):
            # 过滤全空行
            cells = []
            for cell in row:
                if cell is not None:
                    cells.append(str(cell).strip())
            if cells:
                sheet_rows.append(" | ".join(cells))
        
        if sheet_rows:
            sheets_text.append(f"【工作表: {sheet_name}】\n" + "\n".join(sheet_rows))
    
    wb.close()
    
    if not sheets_text:
        raise ValueError("Excel文件没有可读取的工作表数据（可能全为空表）")
    
    full_text = "\n\n".join(sheets_text)
    logger.info(f"Excel解析完成: {file_path}, 共{len(sheets_text)}个工作表, {len(full_text)}字符")
    return full_text


# 支持的文件格式映射
SUPPORTED_FORMATS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".xlsx": parse_excel,
    ".xls": parse_excel,
}


def parse_document(file_path: str) -> Dict[str, str]:
    """
    根据文件格式自动选择解析器
    返回: {"text": 解析文本, "filename": 文件名}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 检查文件大小（限制50MB）
    file_size = os.path.getsize(file_path)
    if file_size > 50 * 1024 * 1024:
        raise ValueError(f"文件过大（{file_size / 1024 / 1024:.1f}MB），上限50MB")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"不支持的文件格式: {ext}。"
            f"仅支持: {', '.join(SUPPORTED_FORMATS.keys())}"
        )
    
    parser_func = SUPPORTED_FORMATS[ext]
    text = parser_func(file_path)
    filename = os.path.basename(file_path)
    
    return {"text": text, "filename": filename}
