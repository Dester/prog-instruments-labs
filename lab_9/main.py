import csv
import re
import sys
from pathlib import Path
from loguru import logger
from checksum import calculate_checksum, serialize_result


class CSVValidator:
    def __init__(self, filename):
        self.filename = filename
        self.row_numbers = []
        self.setup_logging()

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

        logger.info(f"Инициализация валидатора для файла: {filename}")

    def setup_logging(self):
        logger.remove()

        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True,
        )

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logger.add(
            log_dir / "validator_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
        )

        logger.add(
            log_dir / "errors_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
        )

        logger.debug("Логирование настроено успешно")

    def validate_email(self, value):
        is_valid = bool(re.match(self.patterns["email"], value))
        if not is_valid:
            logger.debug(f"Невалидный email: {value}")
        return is_valid

    def validate_http_status(self, value):
        is_valid = bool(re.match(self.patterns["http_status_message"], value))
        if not is_valid:
            logger.debug(f"Невалидный HTTP статус: {value}")
        return is_valid

    def validate_inn(self, value):
        is_valid = bool(re.match(self.patterns["inn"], value))
        if not is_valid:
            logger.debug(f"Невалидный ИНН: {value}")
        return is_valid

    def validate_passport(self, value):
        is_valid = bool(re.match(self.patterns["passport"], value))
        if not is_valid:
            logger.debug(f"Невалидный паспорт: {value}")
        return is_valid

    def validate_ip(self, value):
        is_valid = bool(re.match(self.patterns["ip_v4"], value))
        if not is_valid:
            logger.debug(f"Невалидный IP-адрес: {value}")
        return is_valid

    def validate_latitude(self, value):
        if not re.match(self.patterns["latitude"], value):
            logger.debug(f"Невалидный формат широты: {value}")
            return False
        try:
            lat = float(value)
            is_valid = -90 <= lat <= 90
            if not is_valid:
                logger.debug(f"Широта вне допустимого диапазона: {value}")
            return is_valid
        except ValueError as e:
            logger.error(f"Ошибка преобразования широты '{value}': {e}")
            return False

    def validate_hex_color(self, value):
        is_valid = bool(re.match(self.patterns["hex_color"], value))
        if not is_valid:
            logger.debug(f"Невалидный HEX-цвет: {value}")
        return is_valid

    def validate_isbn(self, value):
        is_valid = bool(re.match(self.patterns["isbn"], value))
        if not is_valid:
            logger.debug(f"Невалидный ISBN: {value}")
        return is_valid

    def validate_uuid(self, value):
        is_valid = bool(re.match(self.patterns["uuid"], value))
        if not is_valid:
            logger.debug(f"Невалидный UUID: {value}")
        return is_valid

    def validate_time(self, value):
        is_valid = bool(re.match(self.patterns["time"], value))
        if not is_valid:
            logger.debug(f"Невалидное время: {value}")
        return is_valid

    def validate_row(self, row, row_num):
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

        invalid_fields = []

        for i, (validator, value) in enumerate(zip(validators, row)):
            field_name = self.columns_order[i]
            if not validator(value):
                invalid_fields.append(field_name)

        if invalid_fields:
            logger.warning(
                f"Строка {row_num} содержит невалидные поля: {', '.join(invalid_fields)}"
            )
            return False

        logger.debug(f"Строка {row_num} успешно прошла валидацию")
        return True

    def process_file(self):
        logger.info(f"Начало обработки файла: {self.filename}")

        try:
            with open(self.filename, "r", encoding="utf-16") as file:
                csv_reader = csv.reader(file, delimiter=";")

                # Пропускаем заголовок
                try:
                    header = next(csv_reader)
                    logger.info(f"Заголовок файла: {header}")
                except StopIteration:
                    logger.error("Файл пуст или не содержит данных")
                    return []

                processed_rows = 0

                for row_num, row in enumerate(csv_reader):
                    row = [cell.strip('"') for cell in row]

                    if len(row) != len(self.columns_order):
                        logger.error(
                            f"Строка {row_num} содержит {len(row)} полей, ожидалось {len(self.columns_order)}"
                        )
                        self.row_numbers.append(row_num)
                        continue

                    if not self.validate_row(row, row_num):
                        self.row_numbers.append(row_num)

                    processed_rows += 1

                    if processed_rows % 1000 == 0:
                        logger.info(
                            f"Обработано {processed_rows} строк. Найдено невалидных: {len(self.row_numbers)}"
                        )

                logger.success(
                    f"Обработка завершена. Всего строк: {processed_rows}, невалидных: {len(self.row_numbers)}"
                )

        except FileNotFoundError:
            logger.error(f"Файл не найден: {self.filename}")
            raise
        except Exception as e:
            logger.exception(f"Критическая ошибка при обработке файла: {e}")
            raise

        return self.row_numbers


def main():
    VARIANT = 53

    logger.info(f"Запуск валидатора для варианта {VARIANT}")

    try:
        validator = CSVValidator("53.csv")
        invalid_rows = validator.process_file()

        logger.info(f"Вычисление контрольной суммы для {len(invalid_rows)} строк")
        checksum = calculate_checksum(invalid_rows)

        logger.info(f"Контрольная сумма: {checksum}")
        logger.info(f"Сериализация результатов в result.json")

        serialize_result(VARIANT, checksum)

        logger.success("Программа успешно завершена")

    except Exception as e:
        logger.critical(f"Программа завершилась с ошибкой: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

