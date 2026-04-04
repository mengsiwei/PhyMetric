import React, { useState, useMemo } from "react";
import { courses, getCourse, CATEGORY, semesterLabel } from "../data/catalog";
import { arePrerequisitesMet, isOfferedInSemester, getAvailableElectives } from "../utils/planEngine";

export default function CoursePickerModal({ semesterIndex, plan, onAdd, onClose }) {
  const [search, setSearch] = useState("");

  const alreadyPlanned = useMemo(() => new Set(plan.flat()), [plan]);

  const categorized = useMemo(() => {
    const categories = {};
    const lowerSearch = search.toLowerCase();

    courses
      .filter((c) => !c.hidden)
      .filter((c) => !alreadyPlanned.has(c.id))
      .filter(
        (c) =>
          c.code.toLowerCase().includes(lowerSearch) ||
          c.name.toLowerCase().includes(lowerSearch)
      )
      .forEach((c) => {
        if (!categories[c.category]) categories[c.category] = [];

        const prereqsMet = arePrerequisitesMet(c.id, semesterIndex, plan).met;
        const offered = isOfferedInSemester(c.id, semesterIndex);

        categories[c.category].push({
          ...c,
          available: prereqsMet && offered,
          reason: !prereqsMet
            ? "Prerequisites not met"
            : !offered
            ? `Not offered in ${semesterIndex % 2 === 0 ? "Fall" : "Spring"}`
            : "",
        });
      });

    // Sort each category: available first
    Object.values(categories).forEach((arr) =>
      arr.sort((a, b) => (b.available ? 1 : 0) - (a.available ? 1 : 0))
    );

    return categories;
  }, [search, alreadyPlanned, semesterIndex, plan]);

  // Also show dynamic elective suggestions
  const dynamicElectives = useMemo(
    () => getAvailableElectives(semesterIndex, plan),
    [semesterIndex, plan]
  );

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal fade-in" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          &#x2715;
        </button>
        <h2>Add Course</h2>
        <p className="modal-subtitle">
          Adding to: {semesterLabel(semesterIndex)}
        </p>

        <input
          className="course-picker-search"
          placeholder="Search courses..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          autoFocus
        />

        {dynamicElectives.length > 0 && !search && (
          <div className="picker-category">
            <div className="picker-category-title" style={{ color: "var(--accent-green)" }}>
              Recommended Technical Electives (prerequisites met & offered this semester)
            </div>
            {dynamicElectives.map((c) => (
              <div
                key={c.id}
                className="picker-course"
                onClick={() => onAdd(semesterIndex, c.id)}
              >
                <div>
                  <span className="picker-course-code">{c.code}</span>
                  {c.name}
                </div>
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{c.credits}cr</span>
              </div>
            ))}
          </div>
        )}

        {Object.entries(categorized).map(([cat, courses]) => (
          <div key={cat} className="picker-category">
            <div className="picker-category-title">{cat}</div>
            {courses.map((c) => (
              <div
                key={c.id}
                className={`picker-course ${!c.available ? "disabled" : ""}`}
                onClick={() => c.available && onAdd(semesterIndex, c.id)}
                title={c.reason || c.description}
              >
                <div>
                  <span className="picker-course-code">{c.code}</span>
                  {c.name}
                  {!c.available && (
                    <span style={{ fontSize: 11, color: "var(--accent-red)", marginLeft: 8 }}>
                      ({c.reason})
                    </span>
                  )}
                </div>
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{c.credits}cr</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
