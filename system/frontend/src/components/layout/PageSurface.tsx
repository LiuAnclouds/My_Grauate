import { ReactNode } from "react";

export function PageSurface({ children, sidebar }: { children: ReactNode; sidebar?: ReactNode }) {
  return (
    <section className={sidebar ? "app-page-grid has-sidebar" : "app-page-grid"}>
      <div className="app-page-primary">{children}</div>
      {sidebar ? <aside className="app-page-secondary">{sidebar}</aside> : null}
    </section>
  );
}
