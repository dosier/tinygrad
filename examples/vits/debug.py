DEB = False

def debug(x, name, full=False):
  if not DEB: return
  nump = x.numpy()
  print(f"{name}({x.shape}, mean={nump.mean():.3f} min={nump.min():.3f}, max={nump.max():.3f})={nump if full else nump.sum()}")
