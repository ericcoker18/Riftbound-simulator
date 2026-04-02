import random


class Deck:
    def __init__(self, cards):
        self.cards = list(cards)

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        if not self.cards:
            return None
        return self.cards.pop()

    def __len__(self):
        return len(self.cards)