import operator 
import itertools

data = [x for x in range(1, 10)]
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

# result = itertools.accumulate(data, operator.mul)
# print(list(result))

# letters_combined = itertools.accumulate(letters, operator.add)
# print(list(letters_combined))

# combinations = itertools.combinations(letters, 5)
# print(list(combinations))

# chained = itertools.chain(data, letters)
# [print(x) for x in chained]

from collections import Counter


# counted = Counter(letters)
# print(counted)
# counter_dict = dict(counted)
# print(counter_dict)
from collections import defaultdict,OrderedDict,deque
# count = defaultdict(int)
# names = "John Julie Jack Ann Mike John John Jack Jack Jen Smith Jen Jen"
# for name in names.split(" "):
#     print(name)
#     count[name] += 1
    
# print(count)

# list = ["a","c","c","a","b","a","a","b","c"]
# cnt = Counter(list)
# ord = OrderedDict(cnt.most_common())
# for key, value in ord.items():
#     print(key, value)
lets = ["a","b","c"]
deq = deque(lets)
print(deq)
deq.append("d")
deq.appendleft("e")
print(deq)
deq.extend(["f","g"])
print(list(deq))
# list = deq.rotate(2)
# print(list)