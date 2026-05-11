import builtins
import ast


class Parameter:
    def __init__(self, name: str, value: str, parameter_type: str):
        self.name = name
        self.value = value
        self.parameter_type = parameter_type

    def to_json(self):
        return {
            "name": self.name,
            "value": str(self.value),
            "parameter_type": self.parameter_type,
        }

    @staticmethod
    def _convert_type(stringed_value: str, parameter_type: str):

        try:
            converted_value = ast.literal_eval(stringed_value)
        except ValueError:
            converted_value = stringed_value
        target_type = getattr(builtins, parameter_type)
        if not isinstance(converted_value, target_type):
            # Convert to target type if needed
            if parameter_type == "float":
                converted_value = float(stringed_value)
            elif parameter_type == "int":
                converted_value = int(stringed_value)

        # Validate final type
        target_type = getattr(builtins, parameter_type)
        if not isinstance(converted_value, target_type):
            raise ValueError(
                f"Type inference went wrong: expected type {target_type} but got {type(converted_value)}"
            )
        return converted_value

    @classmethod
    def from_json(cls, json_data):
        converted_value = cls._convert_type(
            json_data["value"], json_data["parameter_type"]
        )
        return cls(json_data["name"], converted_value, json_data["parameter_type"])
