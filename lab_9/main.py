import csv
import os
import re
import sys
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from checksum import calculate_checksum, serialize_result


class ConfigLoader:
    @staticmethod
    def load_config(env: str = "dev") -> DictConfig:
        base_config = OmegaConf.load("config/config.yaml")

        env_config_path = Path(f"config/config.{env}.yaml")
        if env_config_path.exists():
            env_config = OmegaConf.load(env_config_path)
            config = OmegaConf.merge(base_config, env_config)
        else:
            config = base_config

        schema_path = Path("config/schema.yaml")
        if schema_path.exists():
            schema = OmegaConf.load(schema_path)
            try:
                OmegaConf.merge(schema, config)
                logger.info("Конфигурация прошла валидацию по схеме")
            except Exception as e:
                logger.error(f"Ошибка валидации конфигурации: {e}")
                raise

        return config


class CSVValidator:
    def __init__(self, config: DictConfig):
        self.config = config
        self.filename = config.variant.csv_file
        self.row_numbers = []
        self.setup_logging()

        self.patterns = {
            "email": config.validation_patterns.email,
            "http_status_message": config.validation_patterns.http_status_message,
            "inn": config.validation_patterns.inn,
            "passport": config.validation_patterns.passport,
            "ip_v4": config.validation_patterns.ip_v4,
            "latitude": config.validation_patterns.latitude,
            "hex_color": config.validation_patterns.hex_color,
            "isbn": config.validation_patterns.isbn,
            "uuid": config.validation_patterns.uuid,
            "time": config.validation_patterns.time,
        }

        self.columns_order = list(config.columns_order)

        logger.info(f"Инициализация валидатора для файла: {self.filename}")
        logger.debug(f"Загруженная конфигурация: {OmegaConf.to_yaml(config)}")

    def setup_logging(self):
        """Настройка логирования из конфига"""
        logger.remove()

        log_config = self.config.logging

        if log_config.console.enabled:
            console_level = log_config.console.get("level", log_config.level)
            logger.add(
                sys.stdout,
                format=log_config.console.format,
                level=console_level,
                colorize=log_config.console.colorize,
            )

        if log_config.file.enabled:
            log_dir = Path(log_config.file.directory)
            log_dir.mkdir(exist_ok=True)

            general = log_config.file.general_log
            logger.add(
                log_dir / general.filename,
                format=general.format,
                level=general.level,
                rotation=general.rotation,
                retention=general.retention,
                compression=general.compression if general.compression else None,
            )

            error = log_config.file.error_log
            logger.add(
                log_dir / error.filename,
                format=error.format,
                level=error.level,
                rotation=error.rotation,
                retention=error.retention,
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
        validate_all = self.config.performance.validate_all_fields

        for i, (validator, value) in enumerate(zip(validators, row)):
            field_name = self.columns_order[i]
            if not validator(value):
                invalid_fields.append(field_name)
                if not validate_all:
                    break

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
            with open(self.filename, "r", encoding=self.config.csv.encoding) as file:
                csv_reader = csv.reader(file, delimiter=self.config.csv.delimiter)

                if self.config.csv.skip_header:
                    try:
                        header = next(csv_reader)
                        logger.info(f"Заголовок файла: {header}")
                    except StopIteration:
                        logger.error("Файл пуст или не содержит данных")
                        return []

                processed_rows = 0
                log_interval = self.config.performance.log_progress_interval

                for row_num, row in enumerate(csv_reader):
                    row = [cell.strip(self.config.csv.quote_char) for cell in row]

                    if len(row) != len(self.columns_order):
                        logger.error(
                            f"Строка {row_num} содержит {len(row)} полей, ожидалось {len(self.columns_order)}"
                        )
                        self.row_numbers.append(row_num)
                        continue

                    if not self.validate_row(row, row_num):
                        self.row_numbers.append(row_num)

                    processed_rows += 1

                    if processed_rows % log_interval == 0:
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
    env = os.getenv("APP_ENV", "dev")

    if len(sys.argv) > 1:
        env = sys.argv[1]

    logger.info(f"Запуск валидатора в окружении: {env}")

    try:
        config = ConfigLoader.load_config(env)

        validator = CSVValidator(config)
        invalid_rows = validator.process_file()

        logger.info(f"Вычисление контрольной суммы для {len(invalid_rows)} строк")
        checksum = calculate_checksum(invalid_rows)

        logger.info(f"Контрольная сумма: {checksum}")
        logger.info(f"Сериализация результатов в {config.variant.result_file}")

        serialize_result(config.variant.number, checksum)

        logger.success("Программа успешно завершена")

    except Exception as e:
        logger.critical(f"Программа завершилась с ошибкой: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
