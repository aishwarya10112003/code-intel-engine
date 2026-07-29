"""A tiny sample module so you can see the AST chunker in action."""

import math


def area_of_circle(radius: float) -> float:
    """Return the area of a circle with the given radius."""
    return math.pi * radius**2


class BankAccount:
    """A very small bank account, for demo purposes."""

    def __init__(self, balance: float = 0.0) -> None:
        self.balance = balance

    def deposit(self, amount: float) -> None:
        """Add money to the account."""
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        """Remove money if there are sufficient funds."""
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount
