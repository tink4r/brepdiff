import sys
import occwl.io
from occwl.solid import Solid
from occwl.compound import Compound

# 现在我们直接从 occwl 导入类来进行检查

def probe_step_file(filepath):
    """
    加载一个 STEP 文件并分析其顶级形状的类型和内容。
    此版本修正了 isinstance 检查，以使用 occwl 包装器类。
    """
    print(f"--- 探查文件: {filepath} ---")
    
    try:
        # shapes 现在是一个 occwl 对象的列表
        shapes = occwl.io.load_step(filepath)
    except Exception as e:
        print(f"Error: 加载文件时出错: {e}")
        return

    if not shapes:
        print("结果: 文件为空或无法读取。")
        return

    print(f"发现 {len(shapes)} 个顶级形状。 (occwl 已自动分解顶层复合体)")
    print("=" * 30)

    # 遍历文件中的每一个顶级形状
    for i, shape in enumerate(shapes):
        print(f"分析 顶级形状 #{i+1}:")

        # --- 1. 判断形状的根本类型 (使用 occwl 类) ---
        
        if isinstance(shape, Compound):
            print("  类型: 复合形状 (occwl.compound.Compound)")
            
            # --- 2. 统计内部实体 ---
            num_solids = len(list(shape.solids()))
            num_shells = len(list(shape.shells()))
            num_faces = len(list(shape.faces()))

            print(f"  内部实体统计:")
            print(f"    - Solids (实体): {num_solids}")
            print(f"    - Shells (壳): {num_shells}")
            print(f"    - Faces (面):  {num_faces}")
            
            if num_solids > 1:
                print("  结论: 这是一个多实体 (multi-solid) 复合形状。")

        elif isinstance(shape, Solid):
            print("  类型: 单一实体 (occwl.solid.Solid)")
            
            # 它本身就是一个实体，所以实体数是1
            num_faces = len(list(shape.faces()))
            
            print(f"  内部实体统计:")
            print(f"    - Solids (实体): 1")
            print(f"    - Faces (面):  {num_faces}")

        else:
            print(f"  类型: 其他未知类型 ({type(shape)})")
            
        print("-" * 20)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python probe_step.py <你的_step_文件路径>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    probe_step_file(file_path)