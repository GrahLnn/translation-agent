import os

import translation_agent as ta


if __name__ == "__main__":
    source_lang, target_lang, country = "English", "Chinese", "China"

    
    text_file = "text/ord.txt"

    with open(text_file, encoding="utf-8") as file:
        source_text = file.read()

    translation = ta.translate(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        country=country,
    )
    translation = translation.replace("<TRANSLATION>", "").replace("</TRANSLATION>", "").replace("</TRANSLATE_THIS>", "").replace("<TRANSLATE_THIS>", "")
    os.makedirs(os.path.join("text"), exist_ok=True)
    translation_output_path = "text/translation.txt"

    with open(translation_output_path, "w", encoding="utf-8") as output_file:
        output_file.write(translation)

    print(f"Translation has been written to {translation_output_path}")
