import React from "react";

const MILESTONES = [
  { id: "first", label: "First Semester Done", check: (p, cs) => cs >= 1, icon: "&#127941;" },
  { id: "halfCredits", label: "50% Credits", check: (p) => p.totalCreditsCompleted >= 62, icon: "&#127942;" },
  { id: "csCore", label: "CS Core Complete", check: (p) => {
    const csg = p.groupProgress.find((g) => g.name === "CS Core");
    return csg && csg.percent === 100;
  }, icon: "&#128187;" },
  { id: "mathDone", label: "Math Complete", check: (p) => {
    const mg = p.groupProgress.find((g) => g.name === "Mathematics");
    return mg && mg.percent === 100;
  }, icon: "&#128202;" },
  { id: "allDone", label: "Graduation Ready!", check: (p) => p.overallPercent === 100, icon: "&#127891;" },
];

export default function Milestones({ progress, completedSemesters }) {
  return (
    <div className="milestones">
      {MILESTONES.map((m) => {
        const unlocked = m.check(progress, completedSemesters);
        return (
          <div key={m.id} className={`milestone ${unlocked ? "unlocked" : ""}`}>
            <span className="milestone-icon" dangerouslySetInnerHTML={{ __html: m.icon }} />
            <span className="milestone-text">{m.label}</span>
          </div>
        );
      })}
    </div>
  );
}
