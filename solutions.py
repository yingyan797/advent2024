import numpy as np
def s0_tree_directories():
    class Dir:
      def __init__(self, nm):
        self.name = nm
        self.size = 0
        self.children = []   

    def getSessions(fn):
      rec = open(fn, "r").read()
      return rec.split("$ ")[1:]

    def getCommandResponses(session):
      lines = session.split('\n')
      if (len(lines) > 2):
        return lines[0], lines[1:len(lines)-1]
      return lines[0], []

    def parseCommand(cmd):
      segs = cmd.split(' ')
      return segs[0], segs[1:]

    def parseResponse(lines):
      sizes = []
      names = []
      for l in lines:
        segs = l.split(' ')
        sizes.append(segs[0])
        names.append(segs[1])
      return len(sizes), sizes, names

    def buildDirTree(fn):
      dirsDepth = []
      depth = 0
      sessions = getSessions(fn)
      cur = Dir("init")
      for s in sessions:
        command, responses = getCommandResponses(s)
        instr, args = parseCommand(command)
        match instr:
          case "cd":
            if args[0] == "..":
              cur = cur.par
              depth-=1
            else:
              newDir = Dir(args[0])
              newDir.par = cur
              if len(dirsDepth) <= depth:
                dirsDepth.append([])
              dirsDepth[depth].append(newDir)
              depth+=1
              cur.children.append(newDir)
              cur = newDir

          case "ls":
            l, sizes,names = parseResponse(responses)
            for i in range(l):
              if (sizes[i] != "dir"):
                cur.size += int(sizes[i])
      return dirsDepth

    def testDirTree(dirsDepth):
      i = 0
      for ly in dirsDepth:
        print(i)
        for dir in ly:
          print(dir.size)
          print(dir.name)
        i+=1

    def calcSize(dirsDepth, lim, space, req):
      if (len(dirsDepth) < 2):
        if (dirsDepth[0][0].size <= lim):
          return dirsDepth[0][0].size
        return 0
      sum = 0
      i = len(dirsDepth)-1
      while i >= 0:
        for d in dirsDepth[i]:
          for c in d.children:
            d.size += c.size
          if d.size <= lim:
            sum += d.size
            print(d.size, d.name)
        i-=1
      need = req - (space - dirsDepth[0][0].size)
      minDirSize = dirsDepth[0][0].size
      if need < 0:
        return 0
      for ly in dirsDepth:
        for d in ly:
          if d.size >= need:
            if d.size < minDirSize:
              minDirSize = d.size
      
      return sum, minDirSize


    # Open file
    dirsDepth = buildDirTree("dir.txt")
    print("sum",calcSize(dirsDepth, 100000, 70000000, 30000000))
    #testDirTree(dirsDepth)
    #print(getSessions("dir.txt"))

def s1_compare_lists(fn="static/s1.txt", part=2):
  l1, l2 = [], []
  def parse_locs(line: str):
    locs = []
    loc = ""
    for c in line:
      if c.isnumeric():
        loc += c
      elif loc:
        locs.append(int(loc))
        loc = ""
    if loc:
      locs.append(int(loc))
    return locs

  with open(fn, "r") as f:
    while True:
      line = f.readline()
      if not line:
        break
      locs = parse_locs(line)
      l1.append(locs[0])
      l2.append(locs[1])
  # l1 = [3,4,2,1,3,3]
  # l2 = [4,3,5,3,9,3]
  l1.sort()
  l2.sort()
  if part == 1:
    dist = 0
    for i in range(min(len(l1), len(l2))):
      dist += abs(l1[i]-l2[i])
    return dist
  else:
    i, j = 0, 0
    similarity = 0
    num = 0
    freq = 0
    while i < len(l1) and j < len(l2):
      if num != l1[i]:
        freq = 0
        num = l1[i]
        while j < len(l2) and l2[j] <= num:
          if l2[j] == num:
            freq += 1
          j += 1
      similarity += num*freq
      i += 1
    return similarity

def s2_gradual_monotonic(fn="static/s2.txt"):
  f = open(fn, "r")
  safe_count = 0
  while True:
    line = f.readline()
    if not line:
      break
    prev = None
    num = ""
    incr = 0
    safe = True
    report = []
    for c in line:
      if c.isnumeric():
        num += c
      elif num:
        num = int(num)
        report.append(num)
        if not safe:
          num = ""
          continue
        if prev is not None:
          if not incr:
            if num == prev:
              safe = False
            incr = 1 if num > prev else -1
          det = incr*(num-prev)
          if det < 1 or det > 3:
            safe = False
        prev = num
        num = ""
    if safe:
      # print(report)
      safe_count += 1   
    else:
      for i in range(len(report)):
        comb = report[:i]+report[i+1:]
        prev = None
        incr = 0
        safe = True
        for num in comb:
          if prev is not None:
            if not incr:
              if num == prev:
                safe = False
                break
              incr = 1 if num > prev else -1
            det = incr*(num-prev)
            if det < 1 or det > 3:
              safe = False
              break
          prev = num
        if safe:
          # print(comb)
          safe_count += 1
          break
  return safe_count

def s3_parse_multiply(fn="static/s3.txt"):
  with open(fn, "r") as f:
    instr = f.read()
  i = 0
  enable = True
  result = 0
  while i < len(instr):
    if instr[i:i+4] == "do()":
      enable = True
    elif  instr[i:i+7] == "don't()":
      enable = False
    if enable and instr[i:i+4] == "mul(":
      i += 4
      m1, v1, m2, v2 = "", False, "", False
      while i < len(instr): 
        c = instr[i]
        if c.isnumeric():
          m1 += c
          i += 1
        elif c == ',':
          m1 = int(m1)
          v1 = True
          i += 1
          break
        else:
          break
      if not v1:
        continue
      while i < len(instr): 
        c = instr[i]
        if c.isnumeric():
          m2 += c
          i += 1
        elif c == ')':
          m2 = int(m2)
          v2 = True
          i += 1
          break
        else:
          break
      if not v2:
        continue
      # print(m1, m2)
      result += m1*m2
    else:
      i += 1
  return result

def s4_word_search(fn="static/s4.txt", part=2):
  def to_num(c):
    match c:
      case "X":
        return 1
      case "M":
        return 2
      case "A":
        return 3
      case "S":
        return 4
      case _:
        return 0
  arr = []
  with open(fn, "r") as f:
    while True:
      line = f.readline().strip()
      if not line:
        break
      arr.append(list(map(to_num, line)))
  arr = np.array(arr)
  word_cnt = 0

  if part == 1:
    targets = np.array([[1,2,3,4], [4,3,2,1]])
    for i in range(arr.shape[0]):
      for j in range(arr.shape[1]-3):
        if np.array_equal(arr[i, j:j+4], targets[0]) or np.array_equal(arr[i, j:j+4], targets[1]):
          # print(i,j)
          word_cnt += 1
    for j in range(arr.shape[1]):
      for i in range(arr.shape[0]-3):
        if np.array_equal(arr[i:i+4, j], targets[0]) or np.array_equal(arr[i:i+4, j], targets[1]):
          # print(i,j)
          word_cnt += 1
    for i in range(arr.shape[0]-3):
      for j in range(arr.shape[1]-3):
        diag = np.array(list(arr[i+s,j+s] for s in range(4)))
        if np.array_equal(diag, targets[0]) or np.array_equal(diag, targets[1]):
          # print(i,j)
          word_cnt += 1
    for i in range(arr.shape[0]-3):
      for j in range(3, arr.shape[1]):
        diag = np.array(list(arr[i+s,j-s] for s in range(4)))
        if np.array_equal(diag, targets[0]) or np.array_equal(diag, targets[1]):
          # print(i,j)
          word_cnt += 1
    return word_cnt
  else:
    targets = np.array([[2,3,4], [4,3,2]])
    for i in range(arr.shape[0]-2):
      for j in range(arr.shape[0]-2):
        diag1 = np.array(list(arr[i+s, j+s] for s in range(3)))
        diag2 = np.array(list(arr[i+s, j+2-s] for s in range(3)))
        if (np.array_equal(diag1, targets[0]) or np.array_equal(diag1, targets[1])) and (np.array_equal(diag2, targets[0]) or np.array_equal(diag2, targets[1])):
          print(i,j)
          word_cnt += 1
    return word_cnt
if __name__ == "__main__":
  print(s4_word_search(part=2))

