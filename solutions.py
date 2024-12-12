import numpy as np
from collections import deque
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
  
def s5_schedule_print(fn="static/s5.txt"):
  with open(fn, "r") as f:
    precedence = []
    switch = False
    correct_mid_sum = 0
    incorrect_mid_sum = 0
    while True:
      line = f.readline()
      if not line:
        break
      if not switch:
        if line == "\n":
          switch = True
          continue
        precedence.append([int(i) for i in line.strip().split("|")])
      else:
        update = [int(i) for i in line.strip().split(",")]
        upd_map = {update[i]: i for i in range(len(update))}
        order = True
        for pi, pj in precedence:
          if pi in upd_map and pj in upd_map:
            if upd_map[pi] >= upd_map[pj]:
              order = False
              break
        if order:
          correct_mid_sum += update[int(len(update)/2)]
        else:
          reorder = [update.pop(0)]
          while update:
            num = update.pop(0)
            constraint = [p[1] for p in filter(lambda p: p[0]==num, precedence)]
            append = True
            for i in range(len(reorder)):
              if reorder[i] in constraint:
                reorder.insert(i, num)
                append = False
                break
            if append:
              reorder.append(num)
          incorrect_mid_sum += reorder[int(len(reorder)/2)]
              
  return correct_mid_sum, incorrect_mid_sum

def s6_map_symbols(fn="static/s6.txt", part=2):
  smap = []
  drns = ['^', ">", "v", "<"]
  loc = [-1, -1, -1]
  with open(fn, "r") as f:
    while True:
      line = f.readline()
      if not line:
        break
      row = []
      for c in line.strip():
        if c == '.':
          row.append(0)
        elif c == '#':
          row.append(1)
        else:
          if c in drns:
            loc[0] = len(smap)
            loc[1] = len(row)
            loc[2] = drns.index(c)
          row.append(2)
      smap.append(row)
  orig_map = np.array(smap, dtype=np.uint8)
  # print(smap, loc)
  injections = []

  def traverse(smap, loc, mark=False, end=True):
    all_locs = []
    while True:
      all_locs.append(np.copy(loc))
      match loc[2]:
        case 0:
          for i in range(loc[0]-1, -1, -1):
            if not smap[i][loc[1]]:
              if mark:
                smap[i][loc[1]] = 2
            elif smap[i][loc[1]] == 1:
              loc[2] = 1
              break
            if not end:
              inject = np.copy(loc)
              loc[0] = i
              injections.append((inject, (loc[0], loc[1])))
            else:
              loc[0] = i
        case 1:
          for j in range(loc[1]+1, smap.shape[1]):
            if not smap[loc[0], j]:
              if mark:
                smap[loc[0], j] = 2
            elif smap[loc[0], j] == 1:
              loc[2] = 2
              break
            if not end:
              inject = np.copy(loc)
              loc[1] = j
              injections.append((inject, (loc[0], loc[1])))
            else:
              loc[1] = j
        case 2:
          for i in range(loc[0]+1, smap.shape[0]):
            if not smap[i, loc[1]]:
              if mark:
                smap[i, loc[1]] = 2
            elif smap[i, loc[1]] == 1:
              loc[2] = 3
              break
            if not end:
              inject = np.copy(loc)
              loc[0] = i
              injections.append((inject, (loc[0], loc[1])))
            else:
              loc[0] = i
              
        case 3:
          for j in range(loc[1]-1, -1, -1):
            if not smap[loc[0], j]:
              if mark:
                smap[loc[0], j] = 2
            elif smap[loc[0], j] == 1:
              loc[2] = 0
              break
            if not end:
              inject = np.copy(loc)
              loc[1] = j
              injections.append((inject, (loc[0], loc[1])))
            else:
              loc[1] = j

      # if prev_drn == loc[2]:
      #   cnt = 0
      #   for i in range(smap.shape[0]):
      #     for j in range(smap.shape[1]):
      #       if smap[i,j] == 2:
      #         cnt += 1
      #   return cnt
      if all_locs[-1][2] == loc[2]:
        if part == 1:
          cnt = 0
          for i in range(smap.shape[0]):
            for j in range(smap.shape[1]):
              if smap[i,j] == 2:
                cnt += 1
          return cnt
        return False
      elif any(map(lambda l: np.array_equal(l, loc), all_locs)):
        return True
  if part == 1:
    return traverse(orig_map, loc, mark=True)
  begin = (loc[0], loc[1])
  print(begin)
  traverse(np.copy(orig_map), loc, end=False)
  traj = set()
  result = set()
  for loc, (i,j) in injections:
    if (i,j) not in traj and orig_map[i, j] == 0:
      smap = np.copy(orig_map)
      smap[i, j] = 1
      traj.add((i,j))
      if traverse(smap, loc, mark=False):
        # smap[i, j] = 7
        # print(num)
        # print(smap)
        result.add((i,j))
  return len(result)

def s7_equation_test(fn="static/s7.txt", part=2):
  def check_equation(nums: list, part):   
    cnt = len(nums)-1   
    target = nums.pop(0)
    eq_queue = [(cnt, target, "")]
    while eq_queue:
      l, prod, ops = eq_queue.pop(0)
      if l == 1:
        if nums[0] == prod:
          print(f"{target} = {str(nums[0])+ops}")
          return target
        continue
      head = nums[l-1]
      if prod % head == 0:
        eq_queue.append((l-1, prod/head, f" * {head}"+ops))
      if part == 2:
        rem = prod - head
        lim = pow(10, len(str(head)))
        if rem >= 0 and rem % lim == 0:
          eq_queue.append((l-1, rem/lim, f" || {head}"+ops))

      eq_queue.append((l-1, prod-head, f" + {head}"+ops))
    return 0


  correct_sum = 0
  with open(fn, "r") as f:
    while True:
      line = f.readline()
      if not line:
        return correct_sum
      nums = []
      num = ""
      for c in line.strip():
        if c.isnumeric():
          num += c
        elif num:
          nums.append(int(num))
          num = ""
      if num:
        nums.append(int(num))
      correct_sum += check_equation(nums, part)

def s8_antennna_map(fn="static/s8.txt"):
  lmap = {}
  rows = 0
  cols = 0
  with open(fn, "r") as f:
    while True:
      line = f.readline()
      if not line:
        break
      if not cols:
        cols = len(line.strip())
      for j in range(cols):
        c = line[j]
        if c.isalnum():
          if c not in lmap:
            lmap[c] = []
          lmap[c].append((rows, j))
      rows += 1
  print(lmap, rows, cols)
  antinodes = set()
  def loc_inbound(x, y):
    if x < 0 or x >= rows:
      return False
    if y < 0 or y >= cols:
      return False
    return True
    
  for sig, locs in lmap.items():
    for i in range(len(locs)-1):
      x1, y1 = locs[i]
      for j in range(i+1, len(locs)):
        x2, y2 = locs[j]
        dx, dy = x2-x1, y2-y1
        t1, t2 = x1, y1
        while True:
          if loc_inbound(t1, t2):
            antinodes.add((t1, t2))
          else:
            break
          t1, t2 = t1-dx, t2-dy

        t1, t2 = x2, y2
        while True:
          if loc_inbound(t1, t2):
            antinodes.add((t1, t2))
          else:
            break
          t1, t2 = t1 + dx, t2 + dy

  print(len(antinodes)) 
  return 0     

def s9_file_compact(fn="static/s9.txt", part=2):
  with open(fn, "r") as f:
    orig = f.read()
  alt_file = True
  file_id = 0
  space = []
  def check(arr):
    checksum = 0
    for i in range(len(arr)):
      checksum += i * arr[i]
    return checksum

  if part == 1:
    for c in orig:
      if alt_file:
        space += [file_id]*int(c)
        file_id += 1
      else:
        space += [-1]*int(c)
      alt_file = not alt_file
    i = 0
    while i < len(space):
      if space[i] >= 0:
        i += 1
        continue
      if space[-1] < 0:
        space.pop()
        continue
      space[i] = space.pop()
      i += 1
    return check(space)
    
  else:
    for c in orig:
      if alt_file:
        space.append([file_id, int(c)])
        file_id += 1
      elif c != '0':
        space.append([-1, int(c)])
      alt_file = not alt_file
    i = len(space)-1
    start = 0
    while start < i:
      if space[i][0] >= 0:
        restart = False
        # print(space[i])
        for j in range(start, i):
          if space[j][0] < 0:
            if not restart:
              start = j
              # print(start)
              restart = True
            if space[j][1] >= space[i][1]:
              space[j][0] = space[i][0]
              space[i][0] = -1
              dl = space[j][1] - space[i][1]
              if dl > 0:
                space[j][1] = space[i][1]
                space.insert(j+1, [-1, dl])
                i += 1
              break
      i -= 1
    space_sum = []
    for entry in space:
      if entry[0] <= 0:
        space_sum += [0]*entry[1]
      else:
        space_sum += [entry[0]]*entry[1]
    # print(len(space_sum))
    return check(space_sum)

def s10_terrain_trailhead(fn="static/s10.txt", part=2):
  with open(fn, "r") as f:
    topo = []
    heads = []
    while True:
      line = f.readline()
      if not line:
        break
      hs = []
      for c in line:
        if c.isnumeric():
          if c == '0':
            heads.append((len(topo), len(hs)))
          hs.append(int(c))
      topo.append(hs)

  topo = np.array(topo, dtype=np.uint8)

  def inbound(i, j):
    if i < 0 or i >= topo.shape[0]:
      return False
    if j < 0 or j >= topo.shape[1]:
      return False
    return True
  def find_score(i0, j0):
    locs = {(i0, j0): 1}
    paths = []
    while True:
      nlocs = {}
      for (i, j), l in locs.items():
        num = topo[i, j]
        if num == 9:
          if part == 1:
            return len(locs)
          return sum(locs.values())
        for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
          if inbound(ni, nj) and topo[ni, nj] == num + 1:
            if (ni, nj) not in nlocs:
              nlocs[(ni, nj)] = l
            else:
              nlocs[(ni, nj)] += l
      if nlocs:
        locs = nlocs 
  
  scores = list(map(lambda loc: find_score(loc[0], loc[1]), heads))
  print(scores)
  return sum(scores)

def s11_stone_transform(fn="static/s11.txt"):
  with open(fn, "r") as f:
    line = f.readline().strip()
    stones = {c: 1 for c in line.split(" ")}

  def add_record(rec, num, cnt):
    if num not in rec:
      rec[num] = cnt
    else:
      rec[num] += cnt
    
  for t in range(75):
    nstones = {}
    for num, cnt in stones.items():
      if num == '0':
        add_record(nstones, '1', cnt)
      elif len(num) % 2 == 0:
        l = int(len(num) / 2)
        sl = num[:l]
        r = l
        while num[r] == '0':
          if r == len(num)-1:
            break
          r += 1
        sr = num[r:]
        add_record(nstones, sl, cnt)
        add_record(nstones, sr, cnt)
      else:
        add_record(nstones, str(int(num)*2024), cnt)
    stones = nstones
    # print(stones)
  return sum(stones.values())

def s12_plot_adjoint(fn="static/s12.txt", part=1):
  with open(fn, "r") as f:
    lines = [l.strip() for l in f.readlines()]
  plots = []  # locs, name, perim
  def inbound(i, j):
    if i < 0:
      return False
    if j < 0:
      return False
    return True
  def check_loc(i, j):
    # print(plots)
    plot_ids = []      
    for (ni, nj, drn) in [(i-1, j, 0), (i, j-1, 2)]:
      if inbound(ni, nj):
        for n in range(len(plots)):
          if (ni, nj) in plots[n][0]:
            plot_ids.append((n, drn+1))
            break
    def plot_add(nd, i, j):
      n, drn = nd
      plots[n][0].append((i,j))
      plots[n][2].remove((i, j, drn))
      for (ni, nj, ndrn) in [(i-1, j, 0), (i+1, j, 1), (i, j-1, 2), (i, j+1, 3)]:
        if ndrn != drn-1:
          plots[n][2].append((ni,nj, ndrn))

    if not plot_ids or lines[i][j] not in [plots[n][1] for n,_ in plot_ids]:
      plots.append([[(i, j)], lines[i][j], [(i-1, j, 0), (i+1, j, 1), (i, j-1, 2), (i, j+1, 3)]])
    elif len(plot_ids) == 1:
      plot_add(plot_ids[0], i, j)
    else:
      (p1, d1), (p2, d2) = plot_ids[0], plot_ids[1]
      if p1 == p2:
        plots[p1][0].append((i,j))
        plots[p1][2].remove((i, j, d1))
        plots[p2][2].remove((i, j, d2))
        plots[p1][2] += [(i+1, j, 1), (i, j+1, 3)]
      elif plots[p1][1] == plots[p2][1]:
        plots[p1][2].remove((i, j, d1))
        plots[p2][2].remove((i, j, d2))
        plots[p1][0] += [(i,j)]+plots[p2][0]
        plots[p1][2] += plots[p2][2]
        plots[p1][2] += [(i+1, j, 1), (i, j+1, 3)]
        plots.pop(p2)

      elif lines[i][j] == plots[p1][1]:
        plot_add(plot_ids[0], i, j)
      else:
        plot_add(plot_ids[1], i, j)   

  for i in range(len(lines)):
    for j in range(len(lines[0])):
      # print(i,j, lines[i][j])
      check_loc(i, j)
  price = 0
  if part == 1:
    for plot in plots:
      # print(plot[1], len(plot[2]))
      price += len(plot[0]) * len(plot[2])
  else:
    for plot in plots:
      sides = [[],[],[],[]]
      while plot[2]:
        i, j, drn = plot[2].pop(0)
        sides[drn].append((i,j))
      sides[0].sort()
      sides[1].sort()
      sides[2].sort(key=lambda loc: (loc[1], loc[0]))
      sides[3].sort(key=lambda loc: (loc[1], loc[0]))
      ns = 4
      for drn in [0,1]:
        for i in range(1, len(sides[drn])):
          if sides[drn][i][0] != sides[drn][i-1][0] or sides[drn][i][1] - sides[drn][i-1][1] > 1:
            ns += 1
      for drn in [2,3]:
        for i in range(1, len(sides[drn])):
          if sides[drn][i][1] != sides[drn][i-1][1] or sides[drn][i][0] - sides[drn][i-1][0] > 1:
            ns += 1
      price += len(plot[0]) * ns
      # print(plot[1], ns)

  return price


if __name__ == "__main__":
  # print(s10_terrain_trailhead("static/s10.txt"))
  # print(s11_stone_transform("static/s11.txt"))
  print(s12_plot_adjoint("static/s12.txt", part=2))

