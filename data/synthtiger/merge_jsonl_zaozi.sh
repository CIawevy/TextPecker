#!/bin/bash
# 输出文件路径
OUTPUT_FILE="/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/data/synthtiger/syth_text_data_zaozi.jsonl"  #替换为你的路径

# 清空输出文件
> "$OUTPUT_FILE"

# 查找并合并文件  #替换为你的路径
find /mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/data/synthtiger -type d -name "syth_multitext_zaozi*" -prune -exec sh -c '
  for DIR do
    ANNOT_FILE="$DIR/annotations.jsonl"
    if [ -f "$ANNOT_FILE" ]; then
      echo "合并: $ANNOT_FILE"
      cat "$ANNOT_FILE" >> "$0"
    else
      echo "跳过: $DIR 无 annotations.jsonl"
    fi
  done
' "$OUTPUT_FILE" {} +

# 完成提示
echo -e "\n合并完成！文件: $OUTPUT_FILE"
echo "总行数: $(wc -l < "$OUTPUT_FILE")"