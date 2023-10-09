import csv

# Имя входного CSV-файла
input_file = 'Pegas1.csv'

# Имя выходного CSV-файла
output_file = 'Pegas-trans-output.csv'

# Открываем входной файл для чтения и выходной файл для записи
with open(input_file, 'r', newline='') as csv_in, open(output_file, 'w', newline='') as csv_out:
    reader = csv.reader(csv_in)
    writer = csv.writer(csv_out)

    # Проходим по строкам входного файла и записываем их в выходной файл, исключая пустые строки
    for row in reader:
        if row:
            writer.writerow(row)

print("Пустые строки удалены. Результат сохранен в", output_file)
