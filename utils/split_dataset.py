"""
数据集划分工具

用于将完整数据集划分为训练集和测试集，支持：
1. 按文件划分（推荐）
2. 按比例划分
3. 分层划分（保持各类别比例）
4. K折交叉验证划分

使用方法:
    python utils/split_dataset.py --source ./data/all --dest ./data --split 0.8
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import random
import json


def get_label_from_filename(filename: str, dataset_type: str = 'mfpt') -> str:
    """根据文件名推断标签"""
    name = filename.lower()
    
    if dataset_type == 'mfpt':
        if 'baseline' in name:
            return 'normal'
        elif 'innerracefault' in name or 'inner' in name:
            return 'inner_race'
        elif 'outerracefault' in name or 'outer' in name:
            return 'outer_race'
    elif dataset_type in ['cwru', 'seu']:
        if 'normal' in name:
            return 'normal'
        elif 'ir' in name:
            return 'inner_race'
        elif 'or' in name:
            return 'outer_race'
        elif 'b' in name:
            return 'ball'
    
    return 'unknown'


def split_files_by_ratio(files: List[Path], 
                         train_ratio: float = 0.8,
                         stratify: bool = True,
                         dataset_type: str = 'mfpt') -> Tuple[List[Path], List[Path]]:
    """
    按比例划分文件
    
    参数:
        files: 文件列表
        train_ratio: 训练集比例
        stratify: 是否分层划分（保持各类别比例）
        dataset_type: 数据集类型 ('mfpt', 'cwru', 'seu')
    
    返回:
        train_files, test_files
    """
    if not stratify:
        # 简单随机划分
        random.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        return files[:split_idx], files[split_idx:]
    
    # 分层划分：按类别分组
    label_to_files = {}
    for file in files:
        label = get_label_from_filename(file.name, dataset_type)
        if label not in label_to_files:
            label_to_files[label] = []
        label_to_files[label].append(file)
    
    train_files = []
    test_files = []
    
    # 对每个类别按比例划分
    for label, label_files in label_to_files.items():
        random.shuffle(label_files)
        split_idx = int(len(label_files) * train_ratio)
        train_files.extend(label_files[:split_idx])
        test_files.extend(label_files[split_idx:])
        
        print(f"  类别 '{label}': {len(label_files)} 个文件 -> "
              f"训练集 {split_idx} 个, 测试集 {len(label_files) - split_idx} 个")
    
    return train_files, test_files


def copy_files(files: List[Path], dest_dir: Path):
    """复制文件到目标目录"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        dest_file = dest_dir / file.name
        shutil.copy2(file, dest_file)


def split_dataset(source_dir: str,
                  dest_dir: str,
                  train_ratio: float = 0.8,
                  stratify: bool = True,
                  dataset_type: str = 'mfpt',
                  seed: int = 42):
    """
    划分数据集
    
    参数:
        source_dir: 源数据目录（包含所有.mat文件）
        dest_dir: 目标目录（将创建train和test子目录）
        train_ratio: 训练集比例
        stratify: 是否分层划分
        dataset_type: 数据集类型
        seed: 随机种子
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    
    # 获取所有.mat文件
    mat_files = sorted(list(source_path.glob('*.mat')))
    print(f"\n在 {source_dir} 中找到 {len(mat_files)} 个.mat文件")
    
    if len(mat_files) == 0:
        raise ValueError("没有找到.mat文件")
    
    # 划分文件
    print(f"\n划分比例: 训练集 {train_ratio:.0%}, 测试集 {1-train_ratio:.0%}")
    print(f"分层划分: {'是' if stratify else '否'}")
    print(f"随机种子: {seed}")
    
    train_files, test_files = split_files_by_ratio(
        mat_files, 
        train_ratio=train_ratio,
        stratify=stratify,
        dataset_type=dataset_type
    )
    
    # 创建目标目录
    train_dir = dest_path / 'train'
    test_dir = dest_path / 'test'
    
    print(f"\n复制文件到目标目录...")
    print(f"  训练集: {train_dir} ({len(train_files)} 个文件)")
    copy_files(train_files, train_dir)
    
    print(f"  测试集: {test_dir} ({len(test_files)} 个文件)")
    copy_files(test_files, test_dir)
    
    # 保存划分信息
    split_info = {
        'source_dir': str(source_path),
        'train_ratio': train_ratio,
        'stratify': stratify,
        'dataset_type': dataset_type,
        'seed': seed,
        'total_files': len(mat_files),
        'train_files': len(train_files),
        'test_files': len(test_files),
        'train_file_list': [f.name for f in train_files],
        'test_file_list': [f.name for f in test_files]
    }
    
    split_info_path = dest_path / 'split_info.json'
    with open(split_info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 划分完成！")
    print(f"  训练集: {train_dir}")
    print(f"  测试集: {test_dir}")
    print(f"  划分信息已保存: {split_info_path}")


def create_kfold_splits(source_dir: str,
                       dest_dir: str,
                       n_splits: int = 5,
                       stratify: bool = True,
                       dataset_type: str = 'mfpt',
                       seed: int = 42):
    """
    创建K折交叉验证划分
    
    参数:
        source_dir: 源数据目录
        dest_dir: 目标目录（将创建fold_0, fold_1, ... 子目录）
        n_splits: 折数
        stratify: 是否分层划分
        dataset_type: 数据集类型
        seed: 随机种子
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    
    # 获取所有.mat文件
    mat_files = sorted(list(source_path.glob('*.mat')))
    print(f"\n在 {source_dir} 中找到 {len(mat_files)} 个.mat文件")
    
    if len(mat_files) < n_splits:
        raise ValueError(f"文件数量 ({len(mat_files)}) 少于折数 ({n_splits})")
    
    # 按类别分组
    if stratify:
        label_to_files = {}
        for file in mat_files:
            label = get_label_from_filename(file.name, dataset_type)
            if label not in label_to_files:
                label_to_files[label] = []
            label_to_files[label].append(file)
        
        # 对每个类别的文件进行洗牌
        for label in label_to_files:
            random.shuffle(label_to_files[label])
    else:
        random.shuffle(mat_files)
    
    print(f"\n创建 {n_splits} 折交叉验证划分...")
    
    # 创建每一折
    kfold_info = {
        'source_dir': str(source_path),
        'n_splits': n_splits,
        'stratify': stratify,
        'dataset_type': dataset_type,
        'seed': seed,
        'folds': []
    }
    
    for fold_idx in range(n_splits):
        print(f"\n创建 Fold {fold_idx}...")
        
        train_files = []
        test_files = []
        
        if stratify:
            # 分层K折
            for label, label_files in label_to_files.items():
                n_files = len(label_files)
                fold_size = n_files // n_splits
                
                test_start = fold_idx * fold_size
                test_end = test_start + fold_size if fold_idx < n_splits - 1 else n_files
                
                test_files.extend(label_files[test_start:test_end])
                train_files.extend(label_files[:test_start] + label_files[test_end:])
        else:
            # 普通K折
            n_files = len(mat_files)
            fold_size = n_files // n_splits
            
            test_start = fold_idx * fold_size
            test_end = test_start + fold_size if fold_idx < n_splits - 1 else n_files
            
            test_files = mat_files[test_start:test_end]
            train_files = mat_files[:test_start] + mat_files[test_end:]
        
        # 创建目录
        fold_dir = dest_path / f'fold_{fold_idx}'
        train_dir = fold_dir / 'train'
        test_dir = fold_dir / 'test'
        
        print(f"  训练集: {len(train_files)} 个文件")
        copy_files(train_files, train_dir)
        
        print(f"  测试集: {len(test_files)} 个文件")
        copy_files(test_files, test_dir)
        
        # 记录信息
        kfold_info['folds'].append({
            'fold': fold_idx,
            'train_files': len(train_files),
            'test_files': len(test_files),
            'train_file_list': [f.name for f in train_files],
            'test_file_list': [f.name for f in test_files]
        })
    
    # 保存划分信息
    kfold_info_path = dest_path / 'kfold_info.json'
    with open(kfold_info_path, 'w', encoding='utf-8') as f:
        json.dump(kfold_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ K折交叉验证划分完成！")
    print(f"  划分信息已保存: {kfold_info_path}")


def main():
    parser = argparse.ArgumentParser(description='数据集划分工具')
    parser.add_argument('--source', type=str, required=True,
                       help='源数据目录（包含所有.mat文件）')
    parser.add_argument('--dest', type=str, required=True,
                       help='目标目录（将创建train和test子目录）')
    parser.add_argument('--split', type=float, default=0.8,
                       help='训练集比例（默认0.8）')
    parser.add_argument('--stratify', action='store_true', default=True,
                       help='是否分层划分（默认True）')
    parser.add_argument('--no-stratify', action='store_false', dest='stratify',
                       help='禁用分层划分')
    parser.add_argument('--dataset-type', type=str, default='mfpt',
                       choices=['mfpt', 'cwru', 'seu'],
                       help='数据集类型（默认mfpt）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认42）')
    parser.add_argument('--kfold', type=int, default=None,
                       help='创建K折交叉验证划分（指定折数）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("数据集划分工具")
    print("=" * 80)
    
    if args.kfold:
        create_kfold_splits(
            source_dir=args.source,
            dest_dir=args.dest,
            n_splits=args.kfold,
            stratify=args.stratify,
            dataset_type=args.dataset_type,
            seed=args.seed
        )
    else:
        split_dataset(
            source_dir=args.source,
            dest_dir=args.dest,
            train_ratio=args.split,
            stratify=args.stratify,
            dataset_type=args.dataset_type,
            seed=args.seed
        )


if __name__ == '__main__':
    # 如果没有命令行参数，显示使用示例
    import sys
    if len(sys.argv) == 1:
        print("=" * 80)
        print("数据集划分工具 - 使用示例")
        print("=" * 80)
        print("\n1. 基本用法（80-20划分）:")
        print("   python utils/split_dataset.py \\")
        print("       --source ./data/mfpt_all \\")
        print("       --dest ./data/mfpt_split")
        
        print("\n2. 自定义划分比例（70-30）:")
        print("   python utils/split_dataset.py \\")
        print("       --source ./data/mfpt_all \\")
        print("       --dest ./data/mfpt_split \\")
        print("       --split 0.7")
        
        print("\n3. K折交叉验证（5折）:")
        print("   python utils/split_dataset.py \\")
        print("       --source ./data/mfpt_all \\")
        print("       --dest ./data/mfpt_kfold \\")
        print("       --kfold 5")
        
        print("\n4. CWRU数据集:")
        print("   python utils/split_dataset.py \\")
        print("       --source ./data/cwru_all \\")
        print("       --dest ./data/cwru_split \\")
        print("       --dataset-type cwru")
        
        print("\n5. 禁用分层划分:")
        print("   python utils/split_dataset.py \\")
        print("       --source ./data/mfpt_all \\")
        print("       --dest ./data/mfpt_split \\")
        print("       --no-stratify")
        
        print("\n" + "=" * 80)
        print("参数说明:")
        print("  --source       源数据目录")
        print("  --dest         目标目录")
        print("  --split        训练集比例（默认0.8）")
        print("  --stratify     分层划分，保持各类别比例（默认启用）")
        print("  --no-stratify  禁用分层划分")
        print("  --dataset-type 数据集类型: mfpt, cwru, seu（默认mfpt）")
        print("  --seed         随机种子（默认42）")
        print("  --kfold        K折交叉验证折数")
        print("=" * 80)
    else:
        main()

