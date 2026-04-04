from graphviz import Digraph
from pathlib import Path


def build_pipeline_graph() -> Digraph:
    # g = Digraph("PhyMetricPipeline", format="png")
    # g.attr(rankdir="LR", bgcolor="white")
    g = Digraph("PhyMetricPipeline", format="png")
    g.attr(rankdir="TB", bgcolor="white")  # Top-to-Bottom 纵向布局

    # 全局样式
    g.attr("node", shape="box", style="filled,rounded", color="#333333",
           fontname="Helvetica", fontsize="10", fillcolor="#f7f7ff")

    # ------------------------
    # Step a - Frame sampling
    # ------------------------
    with g.subgraph(name="cluster_step_a") as a:
        a.attr(label="Step a – Frame sampling", labelloc="t",
               fontsize="12", fontname="Helvetica-Bold",
               style="rounded", color="#4c6fff")
        a.node("QF", "question_v2_with_kb.json\nscenes / questions /\nframe_indices / KB",
               fillcolor="#e6ecff")
        a.node("VD", "video_dir\nprompt-{idx}.mp4",
               fillcolor="#e6ecff")
        a.node("A1", "load_scenes\n构造 scenes 列表\n+ q_fps / q_total")
        a.node("A2", "get_video_info\n(v_fps, v_total)")
        a.node("A3", "map_question_indices\nq_idx → actual_idx\n按相对帧位置映射")
        a.node("A4", "extract_frames\n(center-1, center, center+1)\n→ frame_xxxx.jpg")
        a.node("A5", "frame_map.json\n{q_idx: [frame_paths]}")

        a.edges([
            ("QF", "A1"),
            ("VD", "A2"),
            ("A1", "A3"),
            ("A2", "A3"),
            ("A3", "A4"),
            ("A4", "A5"),
        ])

    # 输出到 Step b
    g.node("FramesOut", "frames_dir/{scene_id}/\nframe_xxxx.jpg + frame_map.json",
           fillcolor="#e6ecff")
    g.edge("A5", "FramesOut")

    # ------------------------
    # Step b - VLM QA
    # ------------------------
    with g.subgraph(name="cluster_step_b") as b:
        b.attr(label="Step b – VLM QA (QwenVLEvaluator)", labelloc="t",
               fontsize="12", fontname="Helvetica-Bold",
               style="rounded", color="#00a676")

        b.node("B1", "读取 frame_map / 单帧\n得到 frame_paths")
        b.node("B2", "build_vlm_prompt\nscene_caption + KB + question\n→ user_text")
        b.node("B3", "QwenVLEvaluator.query\n≤6 图, 最长边 448")
        b.node("B4", "VLM 输出文本\nANSWER / CONFIDENCE / REASON")
        b.node("B5", "parse_vlm_response\n→ pred_str, confidence, reason")
        b.node("B6", "对比 GT(yes/no)\n得到 correct")
        b.node("B7", "qa_results.json\n按 scene_id / qid 存一条")

        b.edges([
            ("B1", "B2"),
            ("B2", "B3"),
            ("B3", "B4"),
            ("B4", "B5"),
            ("B5", "B6"),
            ("B6", "B7"),
        ])

        # Qwen 内部子流程（简略）
        b.node("Q1", "图像预处理\n最多6张, resize≤448", shape="box", fillcolor="#f0fff8")
        b.node("Q2", "构造 messages\n(system prompt + images + text)", shape="box", fillcolor="#f0fff8")
        b.node("Q3", "processor → inputs\nmodel.generate", shape="box", fillcolor="#f0fff8")
        b.node("Q4", "decode → response\n清理 CUDA 缓存", shape="box", fillcolor="#f0fff8")

        b.edge("Q1", "Q2", style="dashed")
        b.edge("Q2", "Q3", style="dashed")
        b.edge("Q3", "Q4", style="dashed")
        b.edge("Q4", "B4", style="dashed")

    g.edge("FramesOut", "B1")

    # ------------------------
    # Step c - Scoring
    # ------------------------
    with g.subgraph(name="cluster_step_c") as c:
        c.attr(label="Step c – Scoring", labelloc="t",
               fontsize="12", fontname="Helvetica-Bold",
               style="rounded", color="#ff8a3d")

        c.node("C1", "展开 all_q\n(phenomenon, difficulty,\n dimensions, correct, confidence)")
        c.node("C2", "整体指标:\n- overall_accuracy\n- difficulty_weighted\n- dimension_weighted\n- combined_weighted(主指标)")
        c.node("C3", "分组统计:\n- per_difficulty\n- per_dimension\n- per_phenomenon\n- confidence(对/错)")
        c.node("C4", "eval_summary.json\n+ 控制台表格输出")

        c.edges([
            ("C1", "C2"),
            ("C2", "C3"),
            ("C3", "C4"),
        ])

    g.node("QAOut", "qa_results.json", fillcolor="#e6ecff")
    g.edge("B7", "QAOut")
    g.edge("QAOut", "C1")

    return g


def main():
    out_dir = Path("output100/compare_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = build_pipeline_graph()
    output_path = out_dir / "phymetric_pipeline_step_abc"
    graph.render(str(output_path), cleanup=True)
    print(f"Saved pipeline PNG to: {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()