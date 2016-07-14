import re

class Util:
    @staticmethod
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

    @staticmethod
    def alphanum_key(s):
        return [Util.tryint(c) for c in re.split('([0-9]+)', s)]

    @staticmethod
    def sort_nicely(l):
        return sorted(l, key=Util.alphanum_key)
