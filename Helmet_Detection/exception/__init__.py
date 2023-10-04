
import os
import sys


def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in Python script: {file_name} at line no. {exc_tb.tb_lineno}\n" \
                    f"Error message : {str(error)}"
    return error_message


class HelmetException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
