import pydantic
from typing import Protocol


class HasState(Protocol):
    """A class that implements the HasState protocol must be able to serialize its
    state into a pydantic model using get_state and deserialize using set_state.
    """

    def get_state(self) -> pydantic.BaseModel:
        """Get a copy of the current state or parameters.
        """
        ...

    def set_state(self, state: pydantic.BaseModel) -> None:
        """Set the state or parameters.
        """
        ...


class InitFromParams(Protocol):
    """A class that implements the InitFromParams protocol can be easily configured from
    the command line using the pydantic_cli package.

    All subclasses of InitFromParams must define a nested class called Params of type
    pydantic.BaseModel, and all inner Params classes should follow good pydantic style
    such as implementing validators.

    Command line arguments are automatically constructed from the inner Params class.

    Example:
        class MyInitFromParams(InitFromParams):
            class Params(pydantic.BaseModel):
                my_field: int = pydantic.Field(..., cli=('-f', '--my-field'))
                my_other_field : str = pydantic.Field(..., default='foo')
    """

    # It is better to have the Protocol declare a @property than an attribute here
    # because this lets us satisfy the Protocol with an ABC that declares the property
    # as abstract. This protocol is still satisfied by classes with simple short_name
    # and description class attributes. See https://stackoverflow.com/a/68339603/1935085
    @property
    def short_name(self) -> str:
        return ""

    @property
    def description(self) -> str:
        return ""

    class Params(pydantic.BaseModel):
        """Inner class defining parameters available at initialization time.
        """
        pass

    # Note that Protocols cannot specify an __init__, so instead we specify a factory
    # method that creates an instance of the class from params. This has the added
    # benefit that classes can design their own __init__ methods without worrying about
    # Params objects.

    @classmethod
    def new_from_params(cls, params: Params, **kwargs) -> "InitFromParams":
        ...


__all__ = [
    "HasState",
    "InitFromParams",
]
