import numpy as np


class BanditArmsRotator:
    def __init__(self, arms, hints):
        self.arms = arms
        self.hints = hints
        self.current_index = 0

    def _rotate_right(self):
        return [self.arms[-1]] + self.arms[:-1]

    def _update_hints(self, arms, hints):
        sorted_indices = np.argsort(arms) + 1
        updated_hints = []

        if len(self.arms) == 3:
            for hint in hints:
                hint = hint.replace("{max}", f"{sorted_indices[2]}")
                hint = hint.replace("{mid}", f"{sorted_indices[1]}")
                hint = hint.replace("{min}", f"{sorted_indices[0]}")
                updated_hints.append(hint)

        elif len(self.arms) == 5:
            for hint in hints:
                hint = hint.replace("{max2}", f"{sorted_indices[4]}")
                hint = hint.replace("{max1}", f"{sorted_indices[3]}")
                hint = hint.replace("{mid}", f"{sorted_indices[2]}")
                hint = hint.replace("{min1}", f"{sorted_indices[1]}")
                hint = hint.replace("{min2}", f"{sorted_indices[0]}")
                updated_hints.append(hint)
        return updated_hints

    def next(self):
        self.arms = self._rotate_right()
        self.current_index += 1
        new_hints = self._update_hints(self.arms, self.hints)

        return self.arms, new_hints

    def get_arms(self):
        return self.arms


def main():
    arms, hints = [10, 40, 70], ["{min}, {mid}, {max}"]
    rotator = BanditArmsRotator(arms, hints)
    print(rotator.next())
    print(rotator.next())
    print(rotator.next())

    arms, hints = [10, 20, 40, 30, 70], ["{min2}, {min1} {mid}, {max1}, {max2}"]
    rotator = BanditArmsRotator(arms, hints)
    print(rotator.next())
    print(rotator.next())
    print(rotator.next())
    print(rotator.next())
    print(rotator.next())


if __name__ == "__main__":
    main()
