---
<%*
/* 模板版本 : 1.2 | 最后更新 : 2025-02-27 */
// 可以通过 https://www.uuidgenerator.net/version4 获取 或 https://www.uuidgenerator.net/api/guid
// 生成UUID (兼容模式) 
let uuid;
try {
    uuid = crypto.randomUUID().trim();
} catch (error) {
    // 备用UUID生成方案
    uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => 
        (c === 'x' ? (Math.random()*16|0) : (Math.random()*4|0 +8)).toString(16)
    );
}

// 定义带完整状态信息的选项
const statusOptions = [
	{ 
		display: "📝 草稿 | 可继续编辑", 
		value: "draft",
		tags: "status/draft",
		hint: "初始创作阶段"
	},
	{
		display: "🚀 已发布 | 对外可见",
		value: "published",
		tags: "status/published",
		hint: "完成并公开的内容"
	},
	{
		display: "🗄️ 已归档 | 只读状态",
		value: "archived",
		tags: "status/archived",
		hint: "历史文档封存"
	}
];

// 状态选择器
const selectedIndex = await tp.system.suggester(
       statusOptions.map(opt => `${opt.display}\n↳ ${opt.hint}`),
       statusOptions.map((_, index) => index),
       true,
       "选择文档状态 : ",
       0
   );

let selectedStatus = selectedIndex !== undefined ? statusOptions[selectedIndex].value : "draft";
let statusTags = selectedIndex !== undefined ? statusOptions[selectedIndex].tags : "#status/draft";
%>
id: <% uuid %>
aliases:
  - <% uuid %>
  - <% tp.file.title %>
title: <% tp.file.title %>
created: <% tp.file.creation_date("YYYY-MM-DDTHH:mm") %>
author: hel10word
status: <% selectedStatus %>
tags: 
  - <% statusTags %>
summary: <% await tp.system.prompt("请输入摘要 (50字内) ", "这是一段简短的摘要 , 描述文档的主要内容") %>
---

# <% tp.file.title %>















---
可使用 [![](https://img.shields.io/badge/Excalidraw-CCCCFF?style=for-the-badge&logo=excalidraw&logoColor=333&logoWidth=20&labelColor=CCCCFF)](https://excalidraw.com/) 工具打开本文的 [原型图文件](../KnowledgeMatrix/ComputerScience/Network/网络数据包封装与传输/attachments/excalidraw.excalidraw)




