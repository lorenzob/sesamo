import time

from itertools import groupby

class Match:
    
    name = None
    confidence = None
    timestamp = None
    location = None

    def __init__(self, name, confidence, timestamp, location=None):
        self.name = name
        self.confidence = confidence
        self.timestamp = timestamp
        self.location = location

    def __str__(self):
         return self.name + "-" + str(self.confidence)

class MultiMatch:

    threshold = 0
    duration = 10  #secondi
    validMatchesThreshold = 3
    matches = []

    def __init__(self, threshold, duration, location=None, validMatchesThreshold=3):
        self.threshold = threshold
        self.duration = duration
        self.location = location
        self.validMatchesThreshold = validMatchesThreshold

    def record(self, match):

        now = time.time()
        if match.confidence >= self.threshold:
            print("Add " + match.name)
            self.matches.append(match)
        else:
            print("Ignoring " + match.name + " - " + str(match.confidence))

    def check(self):
        
        self.expireOldMatches()

        for key, group in groupby(self.matches, lambda x: x.name):
            matchesList = list(group)
            print( "Check: {} is {}".format(key, len(matchesList)) )
            if len(matchesList) >= self.validMatchesThreshold:
                print( "Opening for: {}".format(key))
                self.matches = filter(lambda m: m.name != key, self.matches)
                return (key, len(matchesList))
            else:
                print( "Not enough for: {}".format(key))
                
        return None

    def expireOldMatches(self):
        #print(len(self.matches))
        now = time.time()
        before = len(self.matches)
        self.matches = filter(lambda m: m.timestamp + self.duration > now, self.matches)
        print("expired: " + str(before - len(self.matches)))
        
    def __str__(self):
        return "mm:[" + ", ".join(str(x) for x in self.matches) + "]"

    
if __name__ == '__main__':
    
    mm = MultiMatch(0.9, 3)
    
    #mm.record(Match("tizio", 0.81, 1000))
    #print(mm.check())
    
    #mm.record(Match("tizio", 0.91, 3000))
    #print(mm.check())
    now = time.time()
    
    x = Match("caio", 0.91, now)
    print(x)
    
    mm.record(Match("caio", 0.91, now))
    mm.record(Match("tizio", 0.91, now))
    print("> " + str(mm.check()))
    
    time.sleep(1)
    print("> " + str(mm.check()))
    
    mm.record(Match("tizio", 0.87, now))
    mm.record(Match("tizio", 0.91, now))
    mm.record(Match("tizio", 0.91, now))
    print("> " + str(mm.check()))
    print(mm)

    
    mm.record(Match("tizio", 0.87, now))
    mm.record(Match("tizio", 0.91, now))
    mm.record(Match("tizio", 0.91, now))
    mm.record(Match("tizio", 0.87, now))
    mm.record(Match("tizio", 0.91, now))
    mm.record(Match("tizio", 0.91, now))
    print(mm)
    time.sleep(5)
    print("> " + str(mm.check()))

    print(mm)



    
    
    