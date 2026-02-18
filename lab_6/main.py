import csv
import re
from checksum import calculate_checksum, serialize_result


class CSVValidator:
    def __init__(self, filename):
        self.filename = filename
        self.row_numbers = []

        self.patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "http_status_message": r"^\d{3} [A-Za-z ]+$",
            "inn": r"^\d{12}$",
            "passport": r"^\d{2} \d{2} \d{6}$",
            "ip_v4": r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
            "latitude": r"^-?(?:90(?:\.0+)?|[1-8]?\d(?:\.\d+)?)$",
            "hex_color": r"^#[0-9a-fA-F]{6}$",
            "isbn": r"^(?:\d{1,5}-){1,4}\d{1,7}-\d$|^\d-\d{4}-\d{4}-\d$",
            "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
            "time": r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d{1,6})?$",
        }

        self.columns_order = [
            "email",
            "http_status_message",
            "inn",
            "passport",
            "ip_v4",
            "latitude",
            "hex_color",
            "isbn",
            "uuid",
            "time",
        ]

    def validate_email(self, value):
        return bool(re.match(self.patterns["email"], value))

    def validate_http_status(self, value):
        return bool(re.match(self.patterns["http_status_message"], value))

    def validate_inn(self, value):
        return bool(re.match(self.patterns["inn"], value))

    def validate_passport(self, value):
        return bool(re.match(self.patterns["passport"], value))

    def validate_ip(self, value):
        return bool(re.match(self.patterns["ip_v4"], value))

    def validate_latitude(self, value):
        if not re.match(self.patterns["latitude"], value):
            return False
        try:
            lat = float(value)
            return -90 <= lat <= 90
        except ValueError:
            return False

    def validate_hex_color(self, value):
        return bool(re.match(self.patterns["hex_color"], value))

    def validate_isbn(self, value):
        return bool(re.match(self.patterns["isbn"], value))

    def validate_uuid(self, value):
        return bool(re.match(self.patterns["uuid"], value))

    def validate_time(self, value):
        return bool(re.match(self.patterns["time"], value))

    def validate_row(self, row):
        validators = [
            self.validate_email,
            self.validate_http_status,
            self.validate_inn,
            self.validate_passport,
            self.validate_ip,
            self.validate_latitude,
            self.validate_hex_color,
            self.validate_isbn,
            self.validate_uuid,
            self.validate_time,
        ]

        for i, (validator, value) in enumerate(zip(validators, row)):
            if not validator(value):
                return False
        return True

    def process_file(self):
        with open(self.filename, "r", encoding="utf-16") as file:
            csv_reader = csv.reader(file, delimiter=";")
            next(csv_reader)

            for row_num, row in enumerate(csv_reader):
                row = [cell.strip('"') for cell in row]

                if not self.validate_row(row):
                    self.row_numbers.append(row_num)

        return self.row_numbers


def main():
    VARIANT = 53

    validator = CSVValidator("53.csv")

    invalid_rows = validator.process_file()

    checksum = calculate_checksum(invalid_rows)

    print(f"Найдено невалидных строк: {len(invalid_rows)}")
    print(f"Контрольная сумма: {checksum}")

    serialize_result(VARIANT, checksum)


if __name__ == "__main__":
    main()
