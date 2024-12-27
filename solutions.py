import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from PIL import Image
import copy

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

def s13_price_positioning(fn="static/s13.txt", part=2):
  def cost(dxa, dya, dxb, dyb, x, y):
    # a * dxa + b * dxb = x
    # a * dya + b * dyb = y
    
    # a * dxa + r * (y - a * dya) = x
    if part == 2:
      x+=10000000000000
      y+=10000000000000
    if dyb * dxa - dxb * dya == 0:
      return 0
    
    num_a = dyb*x - dxb*y
    den_a = dxa*dyb - dxb * dya
    if num_a % den_a != 0:
      return 0
    na = int(num_a / den_a)
    num_b = y - na * dya
    if num_b % dyb != 0:
      return 0
    nb = int(num_b / dyb)
    if part == 1:
      if na > 100 or nb > 100:
        return 0
    
    print(na, nb)
    return na * 3 + nb

  with open(fn, "r") as f:
    tot_cost = 0
    while True:
      line = f.readline()
      if not line:
        return tot_cost
      lines = [line, f.readline(), f.readline()]
      nums = []
      for line in lines:
        num = ""
        for c in line:
          if c.isnumeric():
            num += c
          elif num:
            nums.append(int(num))
            num = ""
        if num:
          nums.append(int(num))
      # print(nums)
      tot_cost += cost(nums[0], nums[1], nums[2], nums[3], nums[4], nums[5])
      f.readline()

def s14_robots_movement(fn="static/s14.txt", part=2):
  rows = 103
  cols = 101
  
  with open(fn, "r") as f:
    locs = []
    vels = []
    while True:
      line = f.readline()
      if not line:
        break
      nums = []
      num = ""
      for c in line:
        if c.isnumeric() or c=='-':
          num += c
        elif num:
          nums.append(int(num))
          num = ""
      if num:
        nums.append(int(num))
      locs.append(nums[:2])
      vels.append(nums[2:])
  locs = np.array(locs, dtype=int)
  vels = np.array(vels, dtype=int)
  lim = np.array([cols, rows])
  if part == 1:
    for i in range(100):
      locs = (locs + vels) % lim

    states = np.zeros((rows, cols), dtype=int)
    for i in range(locs.shape[0]):
      states[locs[i,1], locs[i,0]] += 1
    hrows, hcols = int(rows/2), int(cols/2)
    return np.sum(states[:hrows, :hcols])*np.sum(states[:hrows, hcols+1:])*np.sum(states[hrows+1:, :hcols])*np.sum(states[hrows+1:, hcols+1:])
  else:
    for t in range(100000):
      states = np.zeros((rows, cols), dtype=bool)
      for i in range(locs.shape[0]):
        if not states[locs[i,1], locs[i,0]]:
          states[locs[i,1], locs[i,0]] = True
      num = np.sum(states)
      if num == locs.shape[0]:
        im = Image.fromarray(125*states, mode="L")
        im.save(f"robots/{t}.png")
        print(t)
      locs = (locs + vels) % lim

def s15_robot_pushing(fn="static/s15.txt", part=2):
  with open(fn, "r") as f:
    symbs = {"#":1, "O":0, ".":2, "@":3}
    acts = {"^":0, "v":1, "<":2, ">":3}
    smap = []
    actions = []
    rloc = np.zeros(2, dtype=int)
    while True:
      line = f.readline()
      if not line.strip():
        break
      row = []
      for c in line.strip():
        if c == "@":
          rloc[0] = len(smap)
          rloc[1] = len(row)
        row.append(symbs[c])
      smap.append(row)
    while True:
      line = f.readline()
      if not line:
        break
      actions += [acts[c] for c in line.strip()]
  smap = np.array(smap, dtype=np.uint8)
  directions = np.array([[-1,0], [1,0], [0,-1], [0,1]], dtype=int)

  def move(a, loc):
    adj = tuple(loc+directions[a])
    if smap[adj] == 2:
      smap[adj] = 3
      smap[tuple(loc)] = 2
      return loc+directions[a]
    if smap[adj] == 1:
      return loc
    
    if part == 1:
      nloc = loc+directions[a]
      while True:
        s = smap[tuple(nloc)]
        if s == 0:
          nloc += directions[a]
        elif s == 1:
          return loc
        else:
          smap[tuple(nloc)] = 0
          smap[adj] = 3
          smap[tuple(loc)] = 2
          return loc+directions[a]
    else:
      nloc = loc+directions[a]
      if a >= 2:
        while True:
          s = smap[tuple(nloc)]
          if s > 3:
            nloc += directions[a]
          elif s == 1:
            return loc
          else:
            if a == 2:
              smap[loc[0]:loc[0]+1, nloc[1]:loc[1]] = smap[loc[0]:loc[0]+1, nloc[1]+1:loc[1]+1]
            else:
              smap[loc[0]:loc[0]+1, loc[1]+1:nloc[1]+1] = smap[loc[0]:loc[0]+1, loc[1]:nloc[1]]
            smap[tuple(loc)] = 2
            return loc+directions[a]
      else:
        layers = [(loc[1], loc[1])]
        row = adj[0]
        while True:
          l, r = layers[-1]
          nl, nr = l, r
          if smap[row, nl] == 5:
            nl -= 1
          if smap[row, nr] == 4:
            nr += 1
          if np.any(smap[row, nl:nr+1] == 1):
            return loc
          if np.all(smap[row, nl:nr+1] == 2):
            for i in range(len(layers)-1, -1, -1):
              l, r = layers[i]
              row = loc[0] + directions[a][0]*i
              nrow = row+directions[a][0]
              smap[nrow:nrow+1, l:r+1] = smap[row:row+1, l:r+1]
              smap[row:row+1, l:r+1] = 2
            smap[tuple(loc)] = 2
            return loc+directions[a]
          while smap[row, nl] == 2:
            nl += 1
          while smap[row, nr] == 2:
            nr -= 1
          layers.append((nl, nr))
          row += directions[a][0]

  gps = 0
  if part == 1:
    for a in actions:
      rloc = move(a, rloc)

    for r in range(smap.shape[0]):
      for c in range(smap.shape[1]):
        if smap[r,c] == 0:
          gps += 100*r+c
    print(smap)
  else:
    smap = smap.repeat(2, 1)
    for r in range(smap.shape[0]):
      for c in range(0,smap.shape[1],2):
        if smap[r,c] == 0:
          smap[r,c] = 4
          smap[r,c+1] = 5
    rloc[1] *= 2
    smap[tuple(rloc+directions[3])] = 2
    for a in actions:
      rloc = move(a, rloc)
    print(smap)
    for r in range(smap.shape[0]):
      for c in range(smap.shape[1]):
        if smap[r,c] == 4:
          gps += 100*r+c
  return gps 

def s16_maze_shortest(fn="static/s16.txt"):
  with open(fn, "r") as f:
    maze = []
    sp = np.zeros(3, dtype=int)
    ep = np.zeros(2, dtype=int)
    symbs = {"#":-1, ".":0, "S":0, "E":0}
    while True:
      line = f.readline()
      if not line:
        break
      row = []
      for c in line.strip():
        if c == "S":
          sp[0] = len(maze)
          sp[1] = len(row)
          sp[2] = 3
        elif c == "E":
          ep[0] = len(maze)
          ep[1] = len(row)
        row.append([symbs[c]]*4)
      maze.append(row)
  maze = np.array(maze, dtype=int)
  omaze = np.copy(maze[:,:,0])
  directions = np.array([[-1,0,0], [0,-1,0], [1,0,0], [0,1,0]], dtype=int)

  def valid(loc):
    if (loc[0] < 0 or loc[0] >= maze.shape[0] or loc[1] < 0 or loc[1] >= maze.shape[1]):
      return False
    if maze[tuple(loc)] < 0:
      return False
    return True
  class Path:
    def __init__(self, parent, locs:list):
      self.parent = parent
      self.locs = locs

  mid_paths = {}
  def traverse():
    tree = [Path(None, [sp])]
    def add_path(nt, d):
      # print("add", d)
      for i in range(len(tree)):
        if d < maze[tuple(tree[i].locs[-1])]:
          tree.insert(i, nt)
          return
      tree.append(nt)

    while tree:
      # print(len(tree))
      path = tree.pop(0)
      loc = path.locs[-1]
      if np.array_equal(loc[:2], ep):
        return maze[tuple(loc)], path
      adjacent = []
      for i in range(len(directions)):
        nloc = loc + directions[i]
        nloc[2] = i
        if valid(nloc):
          cost = maze[tuple(loc)] + 1
          if abs(loc[2] - i) == 2:
            # if maze[nloc[0], nloc[1], loc[2]] > 0 and maze[nloc[0], nloc[1], loc[2]] <= cost-1:
            #   continue
            # cost += 2000
            # if cost < maze[tuple(nloc)]:
            #   maze[tuple(nloc)] = cost
            #   adjacent.append(nloc)
            # continue
            if path.parent or len(path.locs) > 1:
              continue

          if i != loc[2]:
            cost += 1000
          if maze[tuple(nloc)] == 0 or cost < maze[tuple(nloc)]:
            maze[tuple(nloc)] = cost
            adjacent.append(nloc)
          elif cost == maze[tuple(nloc)]:
            p = path
            c = maze[tuple(p.locs.pop())]
            consist = True
            while True:
              for l in reversed(p.locs):
                if maze[tuple(l)] >= c:
                  consist = False
                  break
                c = maze[tuple(l)]
              if not p.parent:
                break
              p = p.parent
            if consist:
              mid_paths[tuple(nloc)] = copy.deepcopy(path)
      if len(adjacent) == 1:
        path.locs.append(adjacent[0])
        add_path(path, maze[tuple(adjacent[0])])
      else:
        for adj in adjacent:
          add_path(Path(path, [adj]), maze[tuple(adj)])
    
  cost, min_path = traverse()
  tree = [min_path]
  bmap = np.zeros((maze.shape[0], maze.shape[1]), dtype=int)
  while tree:
    path = tree.pop(0)
    p = path
    while True:
      for loc in p.locs:
        bmap[loc[0], loc[1]] = 1
        if tuple(loc) in mid_paths:
          tree.append(mid_paths[tuple(loc)])
      
      if not p.parent:
        break
      cp = p
      p = p.parent
      cp.parent = None
      
  neq = np.sum(bmap)
  bmap = bmap + omaze + 1
  Image.fromarray(100*bmap).show()
  # Image.fromarray(100*omaze+100).show()
  return cost, neq
        
def s17_operating_system(fn="static/s17.txt", part=2):
  with open(fn, "r") as f:
    oregs = []
    program = []
    while True:
      line = f.readline()
      if not line.strip():
        break
      num = ""
      for c in line:
        if c.isnumeric():
          num += c
      oregs.append(int(num))
    oregs.append(0)
    line = f.readline()
    num = ""
    for c in line:
      if c.isnumeric():
        num += c
      elif num:
        program.append(int(num))
        num = ""
    if num:
      program.append(int(num))
  
  def exe(a):
    regs = [r for r in oregs]
    regs[0] = a
    ptr = 0
    output = []
    def combo(v):
      if v < 4:
        return v
      return regs[v-4]
    while ptr < len(program)-1:
      # print([bin(regs[i]) for i in range(3)])
      opcode = program[ptr]
      operand = program[ptr+1]

      match opcode:
        case 0:
          regs[0] = int(regs[0] / pow(2, combo(operand)))
        case 1:
          regs[1] = operand ^ regs[1]
        case 2:
          regs[1] = combo(operand) % 8
        case 3:
          if regs[0] != 0:
            ptr = operand
            continue
        case 4:
          regs[1] = regs[1] ^ regs[2]
        case 5:
          output.append(combo(operand) % 8)
        case 6:
          regs[1] = int(regs[0] / pow(2, combo(operand)))
        case 7:
          regs[2] = int(regs[0] / pow(2, combo(operand)))
      ptr += 2
    return output

  if part == 1:
    return exe(oregs[0])
  else:
    bmap = np.zeros(len(program)*3, dtype=bool)
    loc = 0
    def eval(pmap):
      p = 1
      a = 0
      for i in range(pmap.shape[0]-1, -1, -1):
        if pmap[i]:
          a += p
        p *= 2
      return a
    
    for i in range(len(program)-1, -1, -1):
      tg = program[i]
      pref = eval(bmap[:loc])*8
      for src in range(8):
        out = exe(pref + src)[0]
        if out == tg:
          bmap[loc+2] = src % 2
          bmap[loc+1] = int(src / 2) % 2
          bmap[loc] = int(src / 4) % 2
          break
      loc += 3
  
    a = eval(bmap)
    print(a, exe(a), program, exe(a)==program)
 
def s18_ram_maze(fn="static/s18.txt", part=2):
  sz = 71
  omaze = np.zeros((sz,sz), dtype=int)
  rows = []
  with open(fn, "r") as f:
    for _ in range(1024):
      line = f.readline()
      if not line:
        break
      loc = [int(c) for c in line.strip().split(",")]
      rows.append((loc[1], loc[0]))
      omaze[loc[1], loc[0]] = -1
  
  class Path:
    def __init__(self, parent, locs):
      self.parent = parent
      self.locs = locs

  def traverse(maze):
    tree = [Path(None, [(0,0)])]
    def add_path(nt, d):
      for i in range(len(tree)):
        if d < maze[tree[i].locs[-1]]:
          tree.insert(i, nt)
          return
      tree.append(nt)

    def verify(path):
      smap = np.zeros((sz, sz), dtype=np.uint8)
      p = path
      while True:
        for x, y in p.locs:
          smap[x,y] = 200
        if not p.parent:
          # print(smap)
          Image.fromarray(smap).show()
          break
        p = p.parent

    while tree:
      path = tree.pop(0)
      x, y = path.locs[-1]
      if (x, y) == (sz-1, sz-1):
        verify(path)
        return maze[x,y]    
      adjacent = []
      dist = maze[x, y]+1
      for nx, ny in [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]:
        if nx >= 0 and nx < maze.shape[0] and ny >= 0 and ny < maze.shape[1] and maze[nx, ny] >= 0:
          if maze[nx, ny] == 0 or dist < maze[nx, ny]:
            maze[nx, ny] = dist
            adjacent.append((nx, ny))
      
      if len(adjacent) == 1:
        path.locs.append(adjacent[0])
        add_path(path, dist)
      else:
        for adj in adjacent:
          n_path = Path(path, [adj])
          add_path(n_path, dist)
    return -1
  
  if part == 1:
    return traverse(omaze)
  else:
    while rows:
      x, y = rows.pop()
      omaze[x,y] = 0
      if traverse(np.copy(omaze)) >= 0:
        return x,y
    return False

def s19_strip_arrange(fn="static/s19.txt"):
  with open(fn, "r") as f:
    line = f.readline()
    resources = []
    success = []
    elem = ""
    for c in line:
      if c.isalpha():
        elem += c
      elif elem:
        resources.append(elem)
        elem = ""
    if elem:
      resources.append(elem)

    def arrange(target):
      # print(target)
      tree = [0 for _ in range(len(target)+1)]
      tree[0] = 1
      for prog in range(len(target)):
        if not tree[prog]:
          continue
        for i in range(prog+1, len(target)+1):
          if target[prog:i] in resources:
            tree[i] += tree[prog]

      return tree[-1]
          
    f.readline()
    while True:
      line = f.readline()
      if not line:
        print(success)
        return sum(map(lambda entry: entry[1], success))
      target = line.strip()
      strat = arrange(target)
      if strat:
        success.append((target, strat))
  
def s20_maze_cut(fn="static/s20.txt"):
  with open(fn, "r") as f:
    maze = []
    sp = np.zeros(2, dtype=int)
    ep = np.zeros(2, dtype=int)
    symbs = {"#":-1, ".":0, "S":0, "E":0}
    while True:
      line = f.readline()
      if not line:
        break
      row = []
      for c in line.strip():
        if c == "S":
          sp[0] = len(maze)
          sp[1] = len(row)
        elif c == "E":
          ep[0] = len(maze)
          ep[1] = len(row)
        row.append(symbs[c])
      maze.append(row)
  maze = np.array(maze, dtype=int)
  directions = np.array([[-1,0], [0,-1], [0,1], [1,0]])
  loc = sp
  prev = None
  time = 0
  def inbound(l):
    if l[0] >= 0 and l[0] < maze.shape[0] and l[1] >= 0 and l[1] < maze.shape[1]:
      if maze[tuple(l)] >= 0:
        return True
    return False

  track_seq = [sp]
  while True:
    for i in range(4):
      adj = loc+directions[i]
      if inbound(adj) and (prev is None or not np.array_equal(adj, prev)):
        time += 1
        prev = loc
        loc = adj
        track_seq.append(tuple(loc))
        maze[tuple(loc)] = time
        break
    if np.array_equal(loc, ep):
      print(sp, ep, maze[tuple(loc)], len(track_seq))
      break

  count = {}
  def add_cut(dt):
    if dt not in count:
      count[dt] = 1
    else:
      count[dt] += 1
  def cut(lim):
    for i in range(1, maze.shape[0]-1):
      for j in range(1, maze.shape[1]-1):
        if maze[i,j] < 0:
          if maze[i-1,j] >= 0 and maze[i+1,j] >= 0:
            dt1 = abs(maze[i-1,j]-maze[i+1,j])
            if dt1 >= lim:
              add_cut(dt1-2)
          if maze[i,j-1] >= 0 and maze[i,j+1] >= 0:
            dt2 = abs(maze[i,j-1]-maze[i,j+1])
            if dt2 >= lim:
              add_cut(dt2-2)
  
  def cut_multi(lim, pl):
    for i in range(len(track_seq)-lim):
      xf,yf = track_seq[i]
      for j in range(i+lim, len(track_seq)):
        xt,yt = track_seq[j]
        ctime = abs(xt-xf) + abs(yt-yf)
        if ctime <= pl and j >= i+ctime+lim:
          add_cut(j-i-ctime)

  cut_multi(100, 20)
  print(count)
  return sum(count.values())

def s21_robot_transition(fn="static/s21.txt"):
  num_pad = {
    '7': (0,0),'8': (0,1),'9': (0,2),
    '4': (1,0),'5': (1,1),'6': (1,2),
    '1': (2,0),'2': (2,1),'3': (2,2),
               '0': (3,1),'A': (3,2), 
  }
  drn_pad = {
               '^': (0,1),'A': (0,2),
    '<': (1,0),'v': (1,1),'>': (1,2)
  }

  def transit(pad, gap, lf, lt):
    px,py = pad[lf]
    x,y = pad[lt]
    opts = []
    step = ""
    if x == px:
      if y == py:
        step = ''
      elif y > py:
        step = '>'*(y-py)
      elif y < py:
        step = '<'*(py-y)
      opts.append(step+'A')
    elif y == py:
      if x > px:
        step = 'v'*(x-px)
      else:
        step = '^'*(px-x)
      opts.append(step+'A')
    else:
      if (x,py) != gap:
        if x > px:
          step += 'v'*(x-px)
        else:
          step += '^'*(px-x)
        if y > py:
          step += '>'*(y-py)
        else:
          step += '<'*(py-y)
        opts.append(step+'A')
        step = ""
      if (px,y) != gap:
        if y > py:
          step += '>'*(y-py)
        else:
          step += '<'*(py-y)
        if x > px:
          step += 'v'*(x-px)
        else:
          step += '^'*(px-x)
        opts.append(step+'A')
    return opts

  drns = list(drn_pad.keys())
  drn_opts = {}
  for da in drns:
    for db in drns:
      drn_opts[(da, db)] = transit(drn_pad, (0,0), da, db)

  def find_seq(drn_instr, lev):
      ninstr = {}
      for da in drns:
        for db in drns:
          if da == db:
            ninstr[(da, db)] = 1
            continue
          if (da,db) in [("^","v"),("v","^"), ("<",">"), (">", "<")]:
            continue
          opts = drn_opts[(da, db)]
          if lev == 0:
            ninstr[(da,db)] = len(opts[0])
            continue
          instl = 0
          for opt in opts:
            code = 'A'+opt
            stepl = 0
            for i in range(len(code)-1):
              stepl += drn_instr[(code[i+0], code[i+1])]
            if not instl or stepl < instl:
              instl = stepl
          ninstr[(da, db)] = instl
      return ninstr

  drn_instr = {}
  for k in range(25):
    # print(k)
    drn_instr = find_seq(drn_instr, k)

  nums = list(num_pad.keys())
  num_instr = {}
  for na in nums:
    for nb in nums:
      opts = transit(num_pad, (3,0), na, nb)
      instl = 0
      for opt in opts:
        code = 'A'+opt
        stepl = 0
        for i in range(len(code)-1):
          stepl += drn_instr[(code[i], code[i+1])]
        if not instl or stepl < instl:
          instl = stepl
      num_instr[(na, nb)] = instl

  comp = 0
  for src in ["638A","965A","780A","803A","246A"]:
    code = 'A' + src
    instl = 0
    for i in range(len(code)-1):
      instl += num_instr[(code[i], code[i+1])]
      num = int(src[:-1])
    print(instl, num)
    comp += num*instl
  return comp

def s22_pseudo_random(fn="static/s22.txt"):
  def sequence(n, time):
    state = n
    prune_mask = np.power(2, 24)-1
    changes = []
    prices = [n % 10]
    for t in range(time):
      state = ((state << 6) ^ state) & prune_mask
      state = ((state >> 5) ^ state) & prune_mask
      state = ((state << 11) ^ state) & prune_mask
      dig = state % 10
      changes.append(dig-prices[-1])
      prices.append(dig)

    return state, (prices, changes)

  with open(fn, "r") as f:
    sec_sum = np.uint64(0)
    all_signals = {}
    while True:
      line = f.readline()
      if not line:
        # print(all_signals)
        return (sec_sum, max(all_signals.items(), key=lambda kv:kv[1]))
      num = int(line.strip())
      s, (ps, sqs) = sequence(num, 2000)
      sec_sum += s
      signals = {}
      for i in range(len(sqs)-3):
        sig = tuple(sqs[i:i+4])
        p = ps[i+4]
        if sig not in signals:
          signals[sig] = p
        
      for k,v in signals.items():
        if k not in all_signals:
          all_signals[k] = v
        else:
          all_signals[k] += v

def s23_pairwise_interconnect(fn="static/s23.txt"):
  with open(fn, "r") as f:
    matr = []
    nmap = {}
    nlist = []
    def add_pair(n0, n1):
      for name in (n0,n1):
        if name not in nmap:
          nmap[name] = len(nmap)
          nlist.append(name)
          for row in matr:
            row.append(False)
          matr.append([False]*len(nmap))
      matr[nmap[n0]][nmap[n1]] = True
      matr[nmap[n1]][nmap[n0]] = True

    while True:
      line = f.readline()
      if not line:
        break
      names = line.strip().split("-")
      add_pair(names[0], names[1])

  matr = np.array(matr, dtype=bool)
  Image.fromarray(matr*200).show()
  def find_triangles():
    triangles = []
    for r in range(matr.shape[0]):
      valid_r = nlist[r].startswith('t')
      for i in range(r+1, matr.shape[1]-1):
        if not matr[r,i]:
          continue
        valid_i = nlist[i].startswith('t')
        for j in range(i+1, matr.shape[1]):
          if not matr[r,j]:
            continue
          if valid_r or valid_i or nlist[j].startswith('t'):
            if matr[i,j]:
              triangles.append((nlist[r], nlist[i], nlist[j]))
    print(triangles)
    return len(triangles)
  
  def find_fullcon():
    large_group = []
    for r in range(matr.shape[0]):
      group = []
      for i in range(matr.shape[1]):
        if matr[r,i]:
          if group:
            full = True
            for n in group:
              if not matr[i, n]:
                full = False
                break
            if full:
              group.append(i)
          else:
            group.append(i)
      group.append(r)
      if len(group) > len(large_group):
        large_group = group
    if not large_group:
      return ""
    large_group = sorted(list(map(lambda num: nlist[num], large_group)))
    pwd = large_group[0]
    for name in large_group[1:]:
      pwd += ","+name
    return pwd
  return find_fullcon()

def s24_composed_boolean(fn="static/s24.txt"):
  with open(fn, "r") as f:
    wires = {}
    while True:
      line = f.readline()
      if not line.strip():
        break
      segs = line.strip().split(" ")
      wires[segs[0][:-1]] = True if segs[1] == '1' else False
    queue = []
    direct = []
    while True:
      line = f.readline()
      if not line:
        break
      segs = line.strip().split(" ")
      w1, op, w2, w_out = segs[0], segs[1], segs[2], segs[4]
      if(w1.startswith("x") or w1.startswith("y") or w2.startswith("x") or w2.startswith("y")):
        direct.append((w_out, w1, w2, op))
      def _eval(w1, op, w2, w_out):
        in1, in2 = w1 in wires, w2 in wires
        if op == "AND":
          if in1 and in2:
            wires[w_out] = wires[w1] and wires[w2]
          elif (in1 and not wires[w1]) or (in2 and not wires[w2]):
              wires[w_out] = False
        elif op == "OR":
          if in1 and in2:
            wires[w_out] = wires[w1] or wires[w2]
          elif (in1 and wires[w1]) or (in2 and wires[w2]):
            wires[w_out] = True
        elif op == "XOR":
          if in1 and in2:
            wires[w_out] = wires[w1] != wires[w2]
        return w_out in wires

      if not _eval(w1, op, w2, w_out):
        queue.append(([w1, w2],op,w_out))
      else:
        outs = [w_out]
        while outs:
          w_out = outs.pop(0)
          i = 0
          while i < len(queue):
            q_in, q_op, q_out = queue[i]
            if w_out in q_in:
              if _eval(q_in[0], q_op, q_in[1], q_out):
                outs.append(q_out)
                queue.pop(i)
                continue
            i += 1
    def bitsum(symb):
      outs = []
      for k,v in wires.items():
        if k.startswith(symb) and k[1:].isnumeric():
          outs.append((k,v))
      outs.sort()
      bnum = ""
      p = 1
      num = 0
      for n in outs:
        dig = p if n[1] else 0
        num += dig
        bnum = ('1' if n[1] else '0')+bnum
        p *= 2
      return bnum,num
    
    bx, x = bitsum("x")
    by, y = bitsum("y")
    bz, z = bitsum("z")
    rsum = bin(x+y)[2:]
    diff = []
    for i in range(1, len(rsum)+1):
      if rsum[-i] != bz[-i]:
        diff.append(i-1)
    direct.sort(key=lambda entry: min(entry[1][1:],entry[3][1:]))
    # with open("static/ex.txt", "a") as f:
    #   for entry in direct:
    #     f.write(str(entry)+"\n")

    # print(diff)
    # print(bx,by,bz,rsum)
    res = ""
    names = ["z06","ksv","kbs", "nbd","z39","ckb","z20","tqq"]
    names.sort()
    for n in names:
      res += n+"," 
    return res
      
def s25_unique_keylock(fn="static/s25.txt"):
  with open(fn, "r") as f:
    locks = []
    keys = []
    matr = []
    smap = {'#': 1, '.': 0}
    while True:
      for k in range(7):
        line = f.readline().strip()
        matr.append([smap[c] for c in line])
      matr = np.array(matr, dtype=bool)
      if np.all(matr[0]):
        locks.append(np.sum(matr, axis=0)-1)
      else:
        keys.append(np.sum(matr, axis=0)-1)
      matr = []
      if not f.readline():
        break
    print(len(locks), len(keys))
    n_match = 0
    for lock in locks:
      for key in keys:
        if np.any(lock+key > 5):
          continue
        n_match += 1
    return n_match

if __name__ == "__main__":
  # print(s15_robot_pushing("static/s15.txt"))
  # print(s16_maze_shortest("static/s16.txt"))
  # print(s17_operating_system("static/s17.txt", part=2))
  # print(s18_ram_maze("static/s18.txt", part=1))
  # print(s20_maze_cut("static/s20.txt"))
  # print(s21_robot_transition())
  # print(s22_pseudo_random("static/s22.txt"))
  # print(s23_pairwise_interconnect("static/s23.txt"))
  # print(s24_composed_boolean("static/s24.txt"))
  print(s25_unique_keylock("static/s25.txt"))

