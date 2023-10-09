import pandas as pd

# 创建一个空的数据框
combined_df = pd.DataFrame()

# 遍历 x 的范围
for x in range(1, 51):
    # 读取单个表格文件
    file_path = f'out/{x}/detection_results_{x}.xlsx'
    df = pd.read_excel(file_path)

    # 在单个表格开头插入一个带有表格名称的行
    table_name_row = pd.Series([f'Table: detection_results_{x}'])
    df.insert(loc=0, column='Table Name', value=table_name_row)

    # 将单个表格添加到合并的数据框中
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# 保存合并后的数据框为 Excel 文件
combined_df.to_excel('combined_results.xlsx', index=False)
