import pytest
import csv
import json
from unittest.mock import mock_open, patch, MagicMock
from main import CSVValidator, main
from checksum import calculate_checksum, serialize_result


class TestCSVValidator:
    """Тесты для класса CSVValidator"""

    @pytest.fixture
    def validator(self):
        """Фикстура для создания экземпляра валидатора"""
        return CSVValidator("test.csv")

    # Тест 1: Проверка инициализации
    def test_initialization(self, validator):
        """Тест корректной инициализации валидатора"""
        assert validator.filename == "test.csv"
        assert validator.row_numbers == []
        assert len(validator.patterns) == 10
        assert len(validator.columns_order) == 10
        assert validator.columns_order[0] == "email"
        assert validator.columns_order[-1] == "time"

    # Тест 2: Параметризованный тест для email
    @pytest.mark.parametrize(
        "email,expected",
        [
            ("test@email.com", True),
            ("user.name+tag@domain.co.uk", True),
            ("simple@example.com", True),
            ("invalid-email", False),
            ("@domain.com", False),
            ("user@.com", False),
            ("user@domain", False),
            ("", False),
            ("user@domain.c", False),
        ],
    )
    def test_email_validation(self, validator, email, expected):
        """Параметризованный тест валидации email"""
        assert validator.validate_email(email) == expected

    # Тест 3: Параметризованный тест для IP-адресов
    @pytest.mark.parametrize(
        "ip,expected",
        [
            ("192.168.1.1", True),
            ("255.255.255.255", True),
            ("0.0.0.0", True),
            ("127.0.0.1", True),
            ("10.0.0.0", True),
            ("172.16.0.0", True),
            ("256.168.1.1", False),
            ("192.168.1", False),
            ("192.168.1.300", False),
            ("192.168.1.1.1", False),
            ("abc.def.ghi.jkl", False),
            ("", False),
            ("...", False),
        ],
    )
    def test_ip_validation(self, validator, ip, expected):
        """Параметризованный тест валидации IP-адресов"""
        assert validator.validate_ip(ip) == expected

    # Тест 4: Тест с моком для validate_latitude (обработка исключения)
    def test_latitude_validation_with_exception_mock(self, validator):
        """Тест валидации широты с моком для имитации исключения"""
        with patch("builtins.float", side_effect=ValueError("Invalid conversion")):
            result = validator.validate_latitude("not_a_number")
            assert result is False

    # Тест 5: Параметризованный тест для UUID
    @pytest.mark.parametrize(
        "uuid_value,expected",
        [
            ("123e4567-e89b-12d3-a456-426614174000", True),
            ("550e8400-e29b-41d4-a716-446655440000", True),
            ("123e4567-e89b-12d3-a456-42661417400", False),
            ("invalid-uuid", False),
            ("123e4567-e89b-12d3-a456", False),
            ("", False),
            ("gggggggg-gggg-gggg-gggg-gggggggggggg", False),
        ],
    )
    def test_uuid_validation(self, validator, uuid_value, expected):
        """Параметризованный тест валидации UUID"""
        assert validator.validate_uuid(uuid_value) == expected

    # Тест 6: Параметризованный тест для времени
    @pytest.mark.parametrize(
        "time_str,expected",
        [
            ("12:30:45", True),
            ("23:59:59", True),
            ("00:00:00", True),
            ("12:30:45.123", True),
            ("12:30:45.123456", True),
            ("25:00:00", False),
            ("12:60:00", False),
            ("12:30:60", False),
            ("12:30", False),
            ("", False),
            ("12:30:45.", False),
        ],
    )
    def test_time_validation(self, validator, time_str, expected):
        """Параметризованный тест валидации времени"""
        assert validator.validate_time(time_str) == expected

    # Тест 7: Тест с моком для validate_row
    def test_validate_row_with_mocks(self, validator):
        """Тест валидации строки с использованием моков"""
        validator.validate_email = MagicMock(return_value=True)
        validator.validate_http_status = MagicMock(return_value=True)
        validator.validate_inn = MagicMock(return_value=True)
        validator.validate_passport = MagicMock(return_value=True)
        validator.validate_ip = MagicMock(return_value=True)
        validator.validate_latitude = MagicMock(return_value=True)
        validator.validate_hex_color = MagicMock(return_value=True)
        validator.validate_isbn = MagicMock(return_value=True)
        validator.validate_uuid = MagicMock(return_value=True)
        validator.validate_time = MagicMock(return_value=True)

        test_row = [
            "test@email.com",
            "200 OK",
            "123456789012",
            "12 34 567890",
            "192.168.1.1",
            "45.5",
            "#FF5733",
            "978-5-93286-100-8",
            "123e4567-e89b-12d3-a456-426614174000",
            "12:30:45",
        ]

        result = validator.validate_row(test_row)

        assert result is True
        validator.validate_email.assert_called_once_with("test@email.com")
        validator.validate_ip.assert_called_once_with("192.168.1.1")
        validator.validate_time.assert_called_once_with("12:30:45")

    # Тест 8: Параметризованный тест для широты
    @pytest.mark.parametrize(
        "latitude,expected",
        [
            ("45.5", True),
            ("-45.5", True),
            ("0", True),
            ("90", True),
            ("-90", True),
            ("90.0", True),
            ("-90.0", True),
            ("91", False),
            ("-91", False),
            ("45.5.5", False),
            ("abc", False),
            ("", False),
        ],
    )
    def test_latitude_validation(self, validator, latitude, expected):
        """Параметризованный тест валидации широты"""
        assert validator.validate_latitude(latitude) == expected

    # Тест 9: Тест с моком для process_file (С ГЕНЕРАТОРОМ)
    @patch("csv.reader")
    @patch("builtins.open", new_callable=mock_open)
    def test_process_file_with_mock_csv_generator(
        self, mock_file, mock_csv_reader, validator
    ):
        """Тест обработки файла с моком для csv.reader (с генератором)"""
        mock_reader = MagicMock()

        mock_reader.__iter__.return_value = iter(
            [
                [
                    "header1",
                    "header2",
                    "header3",
                    "header4",
                    "header5",
                    "header6",
                    "header7",
                    "header8",
                    "header9",
                    "header10",
                ],
                [
                    "test@email.com",
                    "200 OK",
                    "123456789012",
                    "12 34 567890",
                    "192.168.1.1",
                    "45.5",
                    "#FF5733",
                    "978-5-93286-100-8",
                    "123e4567-e89b-12d3-a456-426614174000",
                    "12:30:45",
                ],
                [
                    "invalid-email",
                    "200 OK",
                    "123456789012",
                    "12 34 567890",
                    "192.168.1.1",
                    "45.5",
                    "#FF5733",
                    "978-5-93286-100-8",
                    "123e4567-e89b-12d3-a456-426614174000",
                    "12:30:45",
                ],
            ]
        )
        mock_csv_reader.return_value = mock_reader

        # Создаем генератор, который всегда возвращает значения
        def validate_generator():
            yield True  # для первой строки
            yield False  # для второй строки
            while True:  # для остальных возможных вызовов
                yield True

        gen = validate_generator()

        with patch.object(
            validator, "validate_row", side_effect=lambda row: next(gen)
        ) as mock_validate:
            result = validator.process_file()

            assert result == [1]
            # Проверяем, что было минимум 2 вызова (фактически 3)
            assert mock_validate.call_count >= 2

    # Тест 10: Тест обработки исключения при открытии файла
    @patch("builtins.open", side_effect=FileNotFoundError("File not found"))
    def test_process_file_not_found(self, mock_open, validator):
        """Тест обработки отсутствия файла"""
        with pytest.raises(FileNotFoundError):
            validator.process_file()


class TestChecksumFunctions:
    """Тесты для функций из модуля checksum"""

    # Тест 11: Тест calculate_checksum
    def test_calculate_checksum(self):
        """Тест вычисления контрольной суммы"""
        result1 = calculate_checksum([3, 1, 2])
        result2 = calculate_checksum([1, 2, 3])
        assert result1 == result2
        assert isinstance(result1, str)
        assert len(result1) == 32

        empty_result = calculate_checksum([])
        assert isinstance(empty_result, str)
        assert len(empty_result) == 32

    # Тест 12: Тест serialize_result с моком
    @patch("builtins.open", new_callable=mock_open)
    def test_serialize_result(self, mock_file):
        """Тест сериализации результата"""
        variant = 53
        checksum = "test_checksum_12345"

        serialize_result(variant, checksum)

        mock_file.assert_called_once_with("result.json", "w", encoding="utf-8")
        handle = mock_file()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)

        assert "variant" in written_data
        assert "53" in written_data
        assert "checksum" in written_data
        assert "test_checksum_12345" in written_data


class TestIntegration:
    """Интеграционные тесты"""

    @pytest.fixture
    def temp_csv_file(self, tmp_path):
        """Фикстура для создания временного CSV файла"""
        file_path = tmp_path / "test.csv"
        with open(file_path, "w", encoding="utf-16") as f:
            f.write(
                "email;http_status_message;inn;passport;ip_v4;latitude;hex_color;isbn;uuid;time\n"
            )
            f.write(
                '"test@email.com";"200 OK";"123456789012";"12 34 567890";"192.168.1.1";"45.5";"#FF5733";"978-5-93286-100-8";"123e4567-e89b-12d3-a456-426614174000";"12:30:45"\n'
            )
            f.write(
                '"invalid-email";"200 OK";"123456789012";"12 34 567890";"192.168.1.1";"45.5";"#FF5733";"978-5-93286-100-8";"123e4567-e89b-12d3-a456-426614174000";"12:30:45"\n'
            )
            f.write(
                '"test@email.com";"200 OK";"123456789012";"12 34 567890";"256.168.1.1";"45.5";"#FF5733";"978-5-93286-100-8";"123e4567-e89b-12d3-a456-426614174000";"12:30:45"\n'
            )
        return str(file_path)

    # Тест 13: Интеграционный тест с временным файлом
    def test_integration_with_real_file(self, temp_csv_file):
        """Интеграционный тест с реальным временным файлом"""
        validator = CSVValidator(temp_csv_file)
        invalid_rows = validator.process_file()

        assert invalid_rows == [1, 2]
        assert len(invalid_rows) == 2


# Тест 14: Тест main функции с моками
@patch("main.serialize_result")
@patch("main.calculate_checksum")
@patch("main.CSVValidator")
def test_main_function(mock_validator_class, mock_calculate, mock_serialize):
    """Тест основной функции с использованием моков"""
    mock_validator = MagicMock()
    mock_validator.process_file.return_value = [1, 2, 3]
    mock_validator_class.return_value = mock_validator

    mock_calculate.return_value = "test_checksum"

    with patch("builtins.print"):
        main()

    mock_validator_class.assert_called_once_with("53.csv")
    mock_validator.process_file.assert_called_once()
    mock_calculate.assert_called_once_with([1, 2, 3])
    mock_serialize.assert_called_once_with(53, "test_checksum")
