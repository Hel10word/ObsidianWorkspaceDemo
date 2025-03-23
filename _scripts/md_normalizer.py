#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import yaml
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("md_normalizer")

# 获取脚本所在目录和项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# 配置文件默认路径
CONFIG_PATH = os.path.join(SCRIPT_DIR, "_config", "md_normalizer.yml")
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "_scripts", "_config", "md_normalizer.yml")

# 默认配置
DEFAULT_CONFIG = {
    "punctuation_mapping": {},
    "punctuation_spacing": {"before_space": False, "after_space": False, "exceptions": {"no_space_after": [], "no_space_before": []}},
    "naming_rules": {"directories": {}, "files": {}, "attachments": {"dirname": "attachments"}},
    "metadata_rules": {"tags": {}},
    "validation_options": {
        "punctuation": True,
        "metadata": True,
        "filenames": True,
        "directories": True,
        "attachments": True,
        "excalidraw": True
    },
    "scan_directories": [],
    "exclude_directories": []
}

@dataclass
class PunctuationIssue:
    """标点符号问题"""
    line_number: int
    position: int
    chinese_punct: str
    english_punct: str
    context: str
    
    def get_description(self):
        return f"第 {self.line_number} 行: 发现中文标点 '{self.chinese_punct}' (应改为 '{self.english_punct}')"

@dataclass
class FilenameIssue:
    """文件名问题"""
    filename: str
    suggested_name: Optional[str] = None
    
    def get_description(self):
        if self.suggested_name:
            return f"文件名 '{self.filename}' 包含空格"
        else:
            return f"文件名 '{self.filename}' 不符合规范"

@dataclass
class DirectoryIssue:
    """目录名问题"""
    dirname: str
    suggested_name: Optional[str] = None
    
    def get_description(self):
        if self.suggested_name:
            return f"目录名 '{self.dirname}' 包含空格"
        else:
            return f"目录名 '{self.dirname}' 不符合规范 (应采用驼峰命名法)"

@dataclass
class AttachmentIssue:
    """附件问题"""
    filename: str
    suggested_name: Optional[str] = None
    
    def get_description(self):
        if self.suggested_name:
            return f"附件文件名 '{self.filename}' 包含空格"
        else:
            return f"附件文件名 '{self.filename}' 不符合规范"

@dataclass
class MetadataIssue:
    """元数据问题"""
    tag: Optional[str] = None
    suggested_tag: Optional[str] = None
    message: str = ""
    
    def get_description(self):
        if self.message:
            return self.message
        elif self.tag and self.suggested_tag:
            return f"标签 '{self.tag}' 应使用小写字母"
        else:
            return "元数据问题"

@dataclass
class ExcalidrawIssue:
    """Excalidraw引用问题"""
    filename: str
    
    def get_description(self):
        return f"Excalidraw文件 '{self.filename}' 在对应的Markdown文件中未被引用"

@dataclass
class ProcessingError:
    """处理错误"""
    error_message: str
    
    def get_description(self):
        return f"处理时出错: {self.error_message}"

@dataclass
class FileValidationResult:
    """文件验证结果"""
    file_path: str
    punctuation_issues: List[PunctuationIssue] = field(default_factory=list)
    filename_issues: List[FilenameIssue] = field(default_factory=list)
    metadata_issues: List[MetadataIssue] = field(default_factory=list)
    excalidraw_issues: List[ExcalidrawIssue] = field(default_factory=list)
    processing_errors: List[ProcessingError] = field(default_factory=list)
    
    def has_issues(self):
        return (len(self.punctuation_issues) > 0 or 
                len(self.filename_issues) > 0 or 
                len(self.metadata_issues) > 0 or 
                len(self.excalidraw_issues) > 0 or 
                len(self.processing_errors) > 0)

@dataclass
class DirectoryValidationResult:
    """目录验证结果"""
    dir_path: str
    directory_issues: List[DirectoryIssue] = field(default_factory=list)
    attachment_issues: List[AttachmentIssue] = field(default_factory=list)
    
    def has_issues(self):
        return len(self.directory_issues) > 0 or len(self.attachment_issues) > 0

@dataclass
class ValidationReport:
    """验证报告类"""
    file_results: Dict[str, FileValidationResult] = field(default_factory=dict)
    directory_results: Dict[str, DirectoryValidationResult] = field(default_factory=dict)
    
    def add_file_result(self, result: FileValidationResult):
        self.file_results[result.file_path] = result
    
    def add_directory_result(self, result: DirectoryValidationResult):
        self.directory_results[result.dir_path] = result
    
    def has_issues(self):
        return (any(result.has_issues() for result in self.file_results.values()) or
                any(result.has_issues() for result in self.directory_results.values()))
    
    def get_all_punctuation_issues(self):
        issues = []
        for file_path, result in self.file_results.items():
            for issue in result.punctuation_issues:
                issues.append((file_path, issue))
        return issues
    
    def get_all_filename_issues(self):
        issues = []
        for file_path, result in self.file_results.items():
            for issue in result.filename_issues:
                issues.append((file_path, issue))
        return issues
    
    def get_all_directory_issues(self):
        issues = []
        for dir_path, result in self.directory_results.items():
            for issue in result.directory_issues:
                issues.append((dir_path, issue))
        return issues
    
    def get_all_attachment_issues(self):
        issues = []
        for dir_path, result in self.directory_results.items():
            for issue in result.attachment_issues:
                issues.append((dir_path, issue))
        return issues
    
    def get_all_metadata_issues(self):
        issues = []
        for file_path, result in self.file_results.items():
            for issue in result.metadata_issues:
                issues.append((file_path, issue))
        return issues
    
    def get_all_excalidraw_issues(self):
        issues = []
        for file_path, result in self.file_results.items():
            for issue in result.excalidraw_issues:
                issues.append((file_path, issue))
        return issues
    
    def get_all_processing_errors(self):
        errors = []
        for file_path, result in self.file_results.items():
            for error in result.processing_errors:
                errors.append((file_path, error))
        return errors
    
    def print_report(self):
        """打印验证报告"""
        if not self.has_issues():
            logger.info("未发现任何问题 , 文档格式已符合规范!")
            return
        
        logger.info("=== Markdown 文档规范化验证报告 ===")
        
        # 统计各类问题数量
        punctuation_issues = self.get_all_punctuation_issues()
        filename_issues = self.get_all_filename_issues()
        directory_issues = self.get_all_directory_issues()
        attachment_issues = self.get_all_attachment_issues()
        metadata_issues = self.get_all_metadata_issues()
        excalidraw_issues = self.get_all_excalidraw_issues()
        processing_errors = self.get_all_processing_errors()
        
        total_issues = (len(punctuation_issues) + len(filename_issues) + 
                        len(directory_issues) + len(attachment_issues) + 
                        len(metadata_issues) + len(excalidraw_issues) + 
                        len(processing_errors))
        
        logger.info(f"总共发现 {total_issues} 个问题:")
        
        # 打印标点符号问题
        if punctuation_issues:
            logger.info(f"\n`标点符号问题`- {len(punctuation_issues)} 个问题:")
            for i, (file_path, issue) in enumerate(punctuation_issues, 1):
                logger.info(f"  {i}. {issue.get_description()}")
                logger.info(f"     文件: {file_path}")
                logger.info(f"     上下文: {issue.context}")
                logger.info(f"     可自动修复: 是")
        
        # 打印文件名问题
        if filename_issues:
            logger.info(f"\n`文件命名问题`- {len(filename_issues)} 个问题:")
            for i, (file_path, issue) in enumerate(filename_issues, 1):
                logger.info(f"  {i}. {issue.get_description()}")
                logger.info(f"     文件: {file_path}")
                if issue.suggested_name:
                    logger.info(f"     建议名称: {issue.suggested_name}")
                    logger.info(f"     可自动修复: 是")
                else:
                    logger.info(f"     可自动修复: 否")
        
        # 打印目录名问题
        if directory_issues:
            logger.info(f"\n`目录命名问题`- {len(directory_issues)} 个问题:")
            for i, (dir_path, issue) in enumerate(directory_issues, 1):
                logger.info(f"  {i}. {issue.get_description()}")
                logger.info(f"     目录: {dir_path}")
                if issue.suggested_name:
                    logger.info(f"     建议名称: {issue.suggested_name}")
                    logger.info(f"     可自动修复: 是")
                else:
                    logger.info(f"     可自动修复: 否")
        
        # 打印附件问题
        if attachment_issues:
            logger.info(f"\n`附件命名问题`- {len(attachment_issues)} 个问题:")
            for i, (dir_path, issue) in enumerate(attachment_issues, 1):
                logger.info(f"  {i}. {issue.get_description()}")
                logger.info(f"     目录: {dir_path}")
                if issue.suggested_name:
                    logger.info(f"     建议名称: {issue.suggested_name}")
                    logger.info(f"     可自动修复: 是")
                else:
                    logger.info(f"     可自动修复: 否")
        
        # 打印元数据问题
        if metadata_issues:
            logger.info(f"\n`元数据问题`- {len(metadata_issues)} 个问题:")
            for i, (file_path, issue) in enumerate(metadata_issues, 1):
                logger.info(f"  {i}. {issue.get_description()}")
                logger.info(f"     文件: {file_path}")
                if issue.tag and issue.suggested_tag:
                    logger.info(f"     可自动修复: 是")
                else:
                    logger.info(f"     可自动修复: 否")
        
        # 打印Excalidraw引用问题
        if excalidraw_issues:
            logger.info(f"\n`Excalidraw引用问题`- {len(excalidraw_issues)} 个问题:")
            for i, (file_path, issue) in enumerate(excalidraw_issues, 1):
                logger.info(f"  {i}. {issue.get_description()}")
                logger.info(f"     文件: {file_path}")
                logger.info(f"     可自动修复: 否")
        
        # 打印处理错误
        if processing_errors:
            logger.info(f"\n`处理错误`- {len(processing_errors)} 个问题:")
            for i, (file_path, error) in enumerate(processing_errors, 1):
                logger.info(f"  {i}. {error.get_description()}")
                logger.info(f"     文件: {file_path}")
                logger.info(f"     可自动修复: 否")
        
        # 统计可自动修复的问题数量
        fixable_count = (len(punctuation_issues) + 
                         len([i for _, i in filename_issues if i.suggested_name]) + 
                         len([i for _, i in directory_issues if i.suggested_name]) + 
                         len([i for _, i in attachment_issues if i.suggested_name]) + 
                         len([i for _, i in metadata_issues if i.tag and i.suggested_tag]))
        
        if fixable_count > 0:
            logger.info(f"\n其中 {fixable_count} 个问题可自动修复 , 使用 --fix 选项来修复它们 .")

class MarkdownNormalizer:
    def __init__(self, config_path=None):
        """初始化规范化工具"""
        # 设置配置文件路径
        self.config_path = config_path if config_path else CONFIG_PATH
        # 设置项目根目录
        self.project_root = PROJECT_ROOT
        # 加载配置
        self.config = DEFAULT_CONFIG.copy()
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    custom_config = yaml.safe_load(f)
                    if custom_config:
                        # 直接使用配置文件 , 而不是合并
                        self.config = custom_config
                        logger.info(f"已加载配置文件: {self.config_path}")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
                logger.info("使用默认配置")
        else:
            logger.warning(f"配置文件不存在: {self.config_path}")
            logger.info("将生成默认配置文件")
            self.save_config(self.config_path)
        
        # 确保配置中有validation_options
        if "validation_options" not in self.config:
            self.config["validation_options"] = DEFAULT_CONFIG["validation_options"]
        
        self.report = ValidationReport()
    
    def save_config(self, config_path):
        """保存当前配置到文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def find_punctuation_issues(self, content, file_path):
        """检查标点符号问题并返回问题列表"""
        if not self.config["validation_options"].get("punctuation", True):
            return []
            
        issues = []
        
        # 创建保护区域
        protected_ranges = []
        
        # 保护区域的函数 , 记录起始和结束行
        def protect_region(pattern, text, name_prefix, is_dotall=False):
            flags = re.DOTALL if is_dotall else 0
            
            for match in re.finditer(pattern, text, flags):
                start_pos = match.start()
                end_pos = match.end()
                
                # 计算区域的起始行号
                lines_before = text[:start_pos].count('\n')
                start_line = lines_before + 1
                
                # 计算区域的结束行号
                matched_text = match.group(0)
                lines_in_match = matched_text.count('\n')
                end_line = start_line + lines_in_match
                
                # 记录保护区域
                protected_ranges.append((start_line, end_line))
        
        # 保护各种特殊区域
        # 1. 首先保护 YAML 元数据块（因为它通常在文件开头）
        yaml_pattern = r'^---\s*\n.*?\n---\s*\n'
        protect_region(yaml_pattern, content, "YAML_BLOCK", True)
        
        # 2. 保护代码块
        code_block_pattern = r'```.*?```'
        protect_region(code_block_pattern, content, "CODE_BLOCK", True)
        
        # 3. 保护内联代码
        inline_code_pattern = r'`[^`\n]+`'
        protect_region(inline_code_pattern, content, "INLINE_CODE")
        
        # 4. 保护 Markdown 链接 [text](url)
        md_link_pattern = r'\[(?:[^\]\\]|\\.)*\]\((?:[^)\\]|\\.)*\)'
        protect_region(md_link_pattern, content, "MD_LINK")
        
        # 5. 保护 Obsidian 内部链接 [[link]]
        obsidian_link_pattern = r'\[\[(?:[^\]\\]|\\.)*\]\]'
        protect_region(obsidian_link_pattern, content, "OB_LINK")
        
        # 6. 保护 HTML 标签
        html_tag_pattern = r'<[^>]+>'
        protect_region(html_tag_pattern, content, "HTML_TAG")
        
        # 获取标点符号映射
        mapping = self.config["punctuation_mapping"]
        
        # 找到所有中文标点符号
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # 检查此行是否在保护区域内
            if any(start <= line_num <= end for start, end in protected_ranges):
                continue
                
            for cn_punct, en_punct in mapping.items():
                if cn_punct in line:
                    # 找到这一行中所有此标点的位置
                    positions = [m.start() for m in re.finditer(re.escape(cn_punct), line)]
                    for pos in positions:
                        # 提取上下文 , 展示问题标点的位置
                        context_start = max(0, pos - 15)
                        context_end = min(len(line), pos + 15)
                        context = line[context_start:context_end]
                        
                        # 高亮问题标点
                        highlighted = context.replace(cn_punct, f"【{cn_punct}】")
                        
                        issues.append(PunctuationIssue(
                            line_number=line_num,
                            position=pos,
                            chinese_punct=cn_punct,
                            english_punct=en_punct,
                            context=highlighted
                        ))
        
        return issues
    
    def fix_punctuation(self, content, issues):
        """根据找到的问题修复标点符号"""
        if not issues:
            return content
            
        # 保护区域的函数
        def protect_regions(content):
            protected_regions = []
            
            # 1. 保护 YAML 元数据块
            yaml_pattern = r'^---\s*\n.*?\n---\s*\n'
            yaml_match = re.search(yaml_pattern, content, re.DOTALL)
            if yaml_match:
                placeholder = "__YAML_BLOCK__"
                protected_regions.append((placeholder, yaml_match.group(0)))
                content = content.replace(yaml_match.group(0), placeholder)
            
            # 2. 保护代码块
            code_block_pattern = r'```.*?```'
            code_blocks = re.finditer(code_block_pattern, content, re.DOTALL)
            for i, match in enumerate(code_blocks):
                placeholder = f"__CODE_BLOCK_{i}__"
                protected_regions.append((placeholder, match.group(0)))
                content = content.replace(match.group(0), placeholder)
            
            # 3. 保护内联代码
            inline_code_pattern = r'`[^`]+`'
            inline_codes = re.finditer(inline_code_pattern, content)
            for i, match in enumerate(inline_codes):
                placeholder = f"__INLINE_CODE_{i}__"
                protected_regions.append((placeholder, match.group(0)))
                content = content.replace(match.group(0), placeholder)
            
            # 4. 保护 Markdown 链接 [text](url)
            md_link_pattern = r'\[(?:[^\]\\]|\\.)*\]\((?:[^)\\]|\\.)*\)'
            md_links = re.finditer(md_link_pattern, content)
            for i, match in enumerate(md_links):
                placeholder = f"__MD_LINK_{i}__"
                protected_regions.append((placeholder, match.group(0)))
                content = content.replace(match.group(0), placeholder)
            
            # 5. 保护 Obsidian 内部链接 [[link]]
            obsidian_link_pattern = r'\[\[(?:[^\]\\]|\\.)*\]\]'
            obsidian_links = re.finditer(obsidian_link_pattern, content)
            for i, match in enumerate(obsidian_links):
                placeholder = f"__OB_LINK_{i}__"
                protected_regions.append((placeholder, match.group(0)))
                content = content.replace(match.group(0), placeholder)
            
            # 6. 保护 HTML 标签
            html_tag_pattern = r'<[^>]+>'
            html_tags = re.finditer(html_tag_pattern, content)
            for i, match in enumerate(html_tags):
                placeholder = f"__HTML_TAG_{i}__"
                protected_regions.append((placeholder, match.group(0)))
                content = content.replace(match.group(0), placeholder)
                
            return content, protected_regions
        
        # 保护特殊区域
        protected_content, protected_regions = protect_regions(content)
        
        # 获取标点符号映射
        mapping = self.config["punctuation_mapping"]
        
        # 获取空格规则配置
        spacing_config = self.config["punctuation_spacing"]
        before_space = spacing_config["before_space"]
        after_space = spacing_config["after_space"]
        no_space_after = spacing_config["exceptions"]["no_space_after"]
        no_space_before = spacing_config["exceptions"]["no_space_before"]
        
        # 替换标点符号
        for cn_punct, en_punct in mapping.items():
            def replace_func(match):
                # 获取前后字符的上下文
                start, end = match.span()
                before = protected_content[start-1] if start > 0 else ''
                after = protected_content[end] if end < len(protected_content) else ''
                
                # 检查是否在列表项中
                # 获取当前行的开始位置
                line_start = protected_content.rfind('\n', 0, start) + 1
                current_line_prefix = protected_content[line_start:start]
                
                # 检查是否匹配列表项格式 (-, *, +, 1., 等)
                is_list_item = re.match(r'^\s*[-*+]|\s*\d+\.\s', current_line_prefix)
                
                # 检查前面的字符是否已经是空格或特殊字符
                prefix = ''
                if before_space and before and before not in [' ', '\n', '\t', '\r'] and en_punct not in no_space_before:
                    # 不在行首添加空格 , 且不在列表项标记后添加额外空格
                    if not (is_list_item and current_line_prefix.endswith(' ')):
                        line_start = protected_content.rfind('\n', 0, start)
                        if line_start == -1 or start - line_start > 1:
                            prefix = ' '
                
                # 检查后面的字符是否已经是空格或特殊字符
                suffix = ''
                if after_space and after and after not in [' ', '\n', '\t', '\r'] and en_punct not in no_space_after:
                    suffix = ' '
                
                return prefix + en_punct + suffix
            
            protected_content = re.sub(re.escape(cn_punct), replace_func, protected_content)
        
        # 处理可能产生的多余空格
        # 1. 替换多个连续空格为单个空格
        # protected_content = re.sub(r' {2,}', ' ', protected_content)
        # 2. 删除行首空格
        # protected_content = re.sub(r'\n +', '\n', protected_content)
        # 3. 删除行尾空格
        protected_content = re.sub(r' +\n', '\n', protected_content)
        # 4. 修复列表项后的多余空格：确保列表项后只有一个空格
        protected_content = re.sub(r'(^\s*[-*+]|\s*\d+\.) +', r'\1 ', protected_content, flags=re.MULTILINE)
        
        # 还原受保护的区域
        for placeholder, original in protected_regions:
            protected_content = protected_content.replace(placeholder, original)
        
        return protected_content
    
    def extract_yaml_metadata(self, content):
        """从Markdown内容中提取YAML元数据"""
        yaml_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.search(yaml_pattern, content, re.DOTALL)
        
        if match:
            yaml_text = match.group(1)
            try:
                metadata = yaml.safe_load(yaml_text)
                if metadata is None:
                    metadata = {}
                return metadata, match.group(0), True
            except yaml.YAMLError:
                return {}, "", False
        
        return {}, "", False
    
    def find_filename_issues(self, file_path):
        """检查文件名问题并返回问题列表"""
        if not self.config["validation_options"].get("filenames", True):
            return []
            
        issues = []
        path = Path(file_path)
        filename = path.name
        
        # 检查文件名中是否有空格
        if self.config["naming_rules"]["files"]["no_spaces"] and " " in filename:
            issues.append(FilenameIssue(
                filename=filename,
                suggested_name=filename.replace(" ", "")
            ))
        
        # 检查文件名格式是否符合规范
        pattern = self.config["naming_rules"]["files"]["pattern"]
        if not re.match(pattern, filename):
            issues.append(FilenameIssue(
                filename=filename
            ))
        
        # 检查特殊规则
        if filename.endswith(".excalidraw"):
            excalidraw_pattern = self.config["naming_rules"]["files"]["special_rules"]["excalidraw"]["pattern"]
            if not re.match(excalidraw_pattern, filename):
                issues.append(FilenameIssue(
                    filename=filename
                ))
        
        return issues
    
    def find_directory_issues(self, dir_path):
        """检查目录名问题并返回问题列表"""
        if not self.config["validation_options"].get("directories", True):
            return []
            
        issues = []
        path = Path(dir_path)
        dirname = path.name
        
        # 跳过根目录
        if not dirname:
            return issues
        
        # 跳过 attachments 目录
        if dirname == self.config["naming_rules"]["attachments"]["dirname"]:
            return issues
        
        # 检查目录名中是否有空格
        if self.config["naming_rules"]["directories"]["no_spaces"] and " " in dirname:
            issues.append(DirectoryIssue(
                dirname=dirname,
                suggested_name=dirname.replace(" ", "")
            ))
        
        # 检查目录名格式是否符合规范
        pattern = self.config["naming_rules"]["directories"]["pattern"]
        if not re.match(pattern, dirname):
            issues.append(DirectoryIssue(
                dirname=dirname
            ))
        
        return issues
    
    def find_attachment_issues(self, attachment_dir):
        """检查附件目录中的文件命名问题并返回问题列表"""
        if not self.config["validation_options"].get("attachments", True):
            return []
            
        issues = []
        
        if not os.path.exists(attachment_dir):
            return issues
        
        for file in os.listdir(attachment_dir):
            file_path = os.path.join(attachment_dir, file)
            if os.path.isfile(file_path):
                # 检查文件名中是否有空格
                if " " in file:
                    issues.append(AttachmentIssue(
                        filename=file,
                        suggested_name=file.replace(" ", "")
                    ))
                
                # 检查文件名格式是否符合规范
                pattern = self.config["naming_rules"]["attachments"]["pattern"]
                if not re.match(pattern, file):
                    issues.append(AttachmentIssue(
                        filename=file
                    ))
        
        return issues
    
    def find_metadata_issues(self, metadata, file_path):
        """检查元数据问题并返回问题列表"""
        if not self.config["validation_options"].get("metadata", True):
            return []
            
        issues = []
        
        if 'tags' not in metadata or not metadata['tags']:
            return issues
        
        tags = metadata['tags']
        if not isinstance(tags, list):
            issues.append(MetadataIssue(
                message="标签必须是列表格式"
            ))
            return issues
        
        tag_rules = self.config["metadata_rules"]["tags"]
        
        for tag in tags:
            # 如果需要小写
            if tag_rules["lowercase"] and any(c.isupper() for c in tag):
                issues.append(MetadataIssue(
                    tag=tag,
                    suggested_tag=tag.lower()
                ))
            
            # 检查单词分隔符
            word_separator = tag_rules["word_separator"]
            if " " in tag:
                issues.append(MetadataIssue(
                    tag=tag,
                    suggested_tag=tag.replace(" ", word_separator)
                ))
        
        return issues
    
    def find_excalidraw_issues(self, md_content, md_file_path):
        """检查Excalidraw引用问题并返回问题列表"""
        if not self.config["validation_options"].get("excalidraw", True):
            return []
            
        issues = []
        filename = os.path.basename(md_file_path)
        
        # 如果是Excalidraw文件 , 检查是否在对应的md文件中被引用
        if filename.endswith(".excalidraw.md"):
            base_name = filename[:-3]  # 移除.md扩展名
            excalidraw_file = os.path.join(os.path.dirname(md_file_path), base_name)
            
            if os.path.exists(excalidraw_file):
                if not f"[[{base_name}]]" in md_content:
                    issues.append(ExcalidrawIssue(
                        filename=base_name
                    ))
        
        return issues
    
    def fix_filename(self, file_path, issues):
        """根据问题修复文件名"""
        if not issues:
            return file_path
        
        # 只处理有建议名称的问题
        fixable_issues = [issue for issue in issues if issue.suggested_name]
        if not fixable_issues:
            return file_path
        
        # 使用第一个可修复问题的建议名称
        new_name = fixable_issues[0].suggested_name
        dir_path = os.path.dirname(file_path)
        new_path = os.path.join(dir_path, new_name)
        
        # 重命名文件
        os.rename(file_path, new_path)
        logger.info(f"已修复文件名: {file_path} -> {new_path}")
        
        return new_path
    
    def fix_directory(self, dir_path, issues):
        """根据问题修复目录名"""
        if not issues:
            return dir_path
        
        # 只处理有建议名称的问题
        fixable_issues = [issue for issue in issues if issue.suggested_name]
        if not fixable_issues:
            return dir_path
        
        # 使用第一个可修复问题的建议名称
        new_name = fixable_issues[0].suggested_name
        parent_dir = os.path.dirname(dir_path)
        new_path = os.path.join(parent_dir, new_name)
        
        # 重命名目录
        os.rename(dir_path, new_path)
        logger.info(f"已修复目录名: {dir_path} -> {new_path}")
        
        return new_path
    
    def fix_attachment(self, attachment_dir, issues):
        """根据问题修复附件文件名"""
        if not issues:
            return
        
        # 只处理有建议名称的问题
        for issue in issues:
            if issue.suggested_name:
                old_path = os.path.join(attachment_dir, issue.filename)
                new_path = os.path.join(attachment_dir, issue.suggested_name)
                
                # 重命名文件
                os.rename(old_path, new_path)
                logger.info(f"已修复附件文件名: {old_path} -> {new_path}")
    
    def fix_metadata(self, content, metadata, issues):
        """根据问题修复元数据"""
        if not issues:
            return content
        
        # 只处理有建议标签的问题
        fixable_issues = [issue for issue in issues if issue.tag and issue.suggested_tag]
        if not fixable_issues:
            return content
        
        # 修改标签
        if 'tags' in metadata and isinstance(metadata['tags'], list):
            for issue in fixable_issues:
                if issue.tag in metadata['tags']:
                    idx = metadata['tags'].index(issue.tag)
                    metadata['tags'][idx] = issue.suggested_tag
        
        # 重新生成YAML块
        yaml_text = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
        yaml_block = f"---\n{yaml_text}---\n"
        
        # 替换原始YAML块
        yaml_pattern = r"^---\s*\n.*?\n---\s*\n"
        new_content = re.sub(yaml_pattern, yaml_block, content, flags=re.DOTALL)
        
        return new_content
    
    def process_markdown_file(self, file_path, fix=False):
        """处理单个Markdown文件 , 进行验证和修正"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 创建文件验证结果对象
            file_result = FileValidationResult(file_path=file_path)
            
            # 检查文件名问题
            filename_issues = self.find_filename_issues(file_path)
            file_result.filename_issues = filename_issues
            
            # 提取和验证YAML元数据
            metadata, yaml_block, has_yaml = self.extract_yaml_metadata(content)
            metadata_issues = []
            if has_yaml:
                metadata_issues = self.find_metadata_issues(metadata, file_path)
                file_result.metadata_issues = metadata_issues
            
            # 检查Excalidraw引用
            excalidraw_issues = self.find_excalidraw_issues(content, file_path)
            file_result.excalidraw_issues = excalidraw_issues
            
            # 检查标点符号问题
            punctuation_issues = self.find_punctuation_issues(content, file_path)
            file_result.punctuation_issues = punctuation_issues
            
            # 添加文件验证结果到报告
            self.report.add_file_result(file_result)
            
            # 如果需要修复问题 , 且确实有问题需要修复
            if fix:
                content_modified = False
                
                # 对于文件名问题 , 只记录警告 , 不自动修复
                if filename_issues:
                    for issue in filename_issues:
                        if issue.suggested_name:
                            logger.warning(f"文件名问题: '{issue.filename}' 建议修改为 '{issue.suggested_name}'")
                        else:
                            logger.warning(f"文件名问题: '{issue.filename}' 不符合命名规范")
                
                # 对于元数据问题 , 只记录警告 , 不自动修复
                if has_yaml and metadata_issues:
                    for issue in metadata_issues:
                        if issue.tag and issue.suggested_tag:
                            logger.warning(f"元数据标签问题: '{issue.tag}' 建议修改为 '{issue.suggested_tag}'")
                        else:
                            logger.warning(issue.get_description())
                
                # 修复标点符号 (这是唯一自动修复的部分) 
                if punctuation_issues:
                    new_content = self.fix_punctuation(content, punctuation_issues)
                    if new_content != content:
                        content = new_content
                        content_modified = True
                        logger.info(f"已修复标点符号: {file_path}")
                        # 打印修改内容摘要
                        logger.info(f"  - 共修复 {len(punctuation_issues)} 处标点符号问题")
                        # 最多显示前5个修改
                        for i, issue in enumerate(punctuation_issues[:5]):
                            logger.info(f"  - 第{issue.line_number}行: '{issue.chinese_punct}' -> '{issue.english_punct}'")
                        if len(punctuation_issues) > 5:
                            logger.info(f"  - ... 以及其他 {len(punctuation_issues) - 5} 处修改")
                
                # 只有当内容确实被修改时 , 才保存文件
                if content_modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                        logger.info(f"已保存修改: {file_path}")
                # else:
                #     logger.info(f"文件内容无需修改: {file_path}")
            
            # 检查附件目录
            file_dir = os.path.dirname(file_path)
            attachments_dir = os.path.join(file_dir, self.config["naming_rules"]["attachments"]["dirname"])
            if os.path.exists(attachments_dir):
                attachment_issues = self.find_attachment_issues(attachments_dir)
                
                # 创建目录验证结果对象
                if attachment_issues:
                    dir_result = DirectoryValidationResult(dir_path=attachments_dir)
                    dir_result.attachment_issues = attachment_issues
                    self.report.add_directory_result(dir_result)
                    
                    # 对于附件问题 , 只记录警告 , 不自动修复
                    if fix and attachment_issues:
                        for issue in attachment_issues:
                            if issue.suggested_name:
                                logger.warning(f"附件文件名问题: '{issue.filename}' 建议修改为 '{issue.suggested_name}'")
                            else:
                                logger.warning(f"附件文件名问题: '{issue.filename}' 不符合命名规范")
        
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            file_result = FileValidationResult(file_path=file_path)
            file_result.processing_errors.append(ProcessingError(error_message=str(e)))
            self.report.add_file_result(file_result)
    
    def process_directory(self, directory, fix=False):
        """递归处理目录中的所有Markdown文件"""
        # 排除指定目录
        base_dir = os.path.basename(directory)
        if base_dir in self.config["exclude_directories"]:
            logger.info(f"跳过排除目录: {directory}")
            return
        
        logger.info(f"处理目录: {directory}")
        
        for root, dirs, files in os.walk(directory):
            # 过滤排除目录
            dirs[:] = [d for d in dirs if d not in self.config["exclude_directories"]]
            
            # 验证目录名
            for d in dirs:
                dir_path = os.path.join(root, d)
                directory_issues = self.find_directory_issues(dir_path)
                
                if directory_issues:
                    # 创建目录验证结果对象
                    dir_result = DirectoryValidationResult(dir_path=dir_path)
                    dir_result.directory_issues = directory_issues
                    self.report.add_directory_result(dir_result)
                    
                    # 对于目录名问题 , 只记录警告 , 不自动修复
                    if fix:
                        for issue in directory_issues:
                            if issue.suggested_name:
                                logger.warning(f"目录名问题: '{issue.dirname}' 建议修改为 '{issue.suggested_name}'")
                            else:
                                logger.warning(f"目录名问题: '{issue.dirname}' 不符合命名规范")
            
            # 处理Markdown文件
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    self.process_markdown_file(file_path, fix)
    
    def run(self, path=None, fix=False):
        """运行规范化工具"""
        if path:
            path = Path(path)
            if path.is_file() and path.suffix == '.md':
                self.process_markdown_file(path, fix)
            elif path.is_dir():
                self.process_directory(path, fix)
            else:
                logger.error(f"错误: {path} 不是有效的Markdown文件或目录")
        else:
            # 处理配置中指定的所有目录
            for dir_name in self.config["scan_directories"]:
                dir_path = os.path.join(self.project_root, dir_name)
                if os.path.exists(dir_path):
                    self.process_directory(dir_path, fix)
                else:
                    logger.warning(f"指定的目录不存在: {dir_path}")
        
        # 输出验证报告
        self.report.print_report()
        
        return self.report

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Markdown文档规范化工具')
    parser.add_argument('path', nargs='?', help='要处理的目录或文件路径（可选）')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--generate-config', help='生成默认配置文件')
    parser.add_argument('--fix', action='store_true', help='自动修复问题')
    args = parser.parse_args()
    
    # 生成默认配置
    if args.generate_config:
        normalizer = MarkdownNormalizer()
        normalizer.save_config(args.generate_config)
        logger.info(f"已生成默认配置文件: {args.generate_config}")
        return
    
    # 初始化工具
    normalizer = MarkdownNormalizer(args.config)
    
    # 运行工具
    normalizer.run(args.path, args.fix)

if __name__ == "__main__":
    main()