import os

import translation_agent as ta


# 设置语言和国家
source_lang, target_lang, country = "English", "Chinese", "China"

# 遍历 text 文件夹中的所有文件
text_folder = "text"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(text_folder):
    text_file_path = os.path.join(text_folder, filename)
    
    # 读取源文本
    with open(text_file_path, encoding="utf-8") as file:
        source_text = file.read()

    # 调用翻译逻辑
    translation = ta.translate(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        country=country,
        curfile=filename,
    )
    translation = (
        translation.replace("<TRANSLATION>", "")
        .replace("</TRANSLATION>", "")
        .replace("</TRANSLATE_THIS>", "")
        .replace("<TRANSLATE_THIS>", "")
    )
    
    # 设置翻译结果输出路径
    translation_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.md")

    # 将翻译结果写入输出文件
    with open(translation_output_path, "w", encoding="utf-8") as output_file:
        output_file.write(translation)
        
    # 删除源文本
    os.remove(text_file_path)

    print(f"Translation has been written to {translation_output_path}")