"""Debug the tree builder on one PDF without the server: prints the tree and LLM usage.

  .venv/bin/python scripts/debug_tree.py samples/earthmover.pdf
"""
import asyncio, logging, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
from pidx import parse, tree, llm

async def main(path):
    t=time.time(); p=parse.parse_pdf(path); t1=time.time()
    nodes, info = await tree.build_tree(p)
    print(path, f"parse {t1-t:.1f}s tree {time.time()-t1:.1f}s", info, llm.usage)
    for n,d in tree.walk_depth(nodes):
        print("  "*d + f"[{n['node_id']}] {n['title'][:70]} ({n['start_index']}-{n['end_index']})")

asyncio.run(main(sys.argv[1]))
