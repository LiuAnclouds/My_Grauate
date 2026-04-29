import { ReactNode } from "react";

type Props = {
  eyebrow: string;
  title: string;
  description: string;
  status?: string;
  meta?: Array<{ label: string; value: string }>;
  actions?: ReactNode;
};

export function PageHeader({ eyebrow, title, description, status, meta = [], actions }: Props) {
  return (
    <section className="app-page-header">
      <div className="app-page-header__main">
        <p className="eyebrow">{eyebrow}</p>
        <div className="app-page-header__title-row">
          <div>
            <h2>{title}</h2>
            <p>{description}</p>
          </div>
          {status ? <span className="app-inline-badge">{status}</span> : null}
        </div>
      </div>
      {meta.length ? (
        <div className="app-page-meta-strip" aria-label="页面上下文信息">
          {meta.map((item) => (
            <div key={item.label} className="app-meta-tile">
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
        </div>
      ) : null}
      {actions ? <div className="app-page-header__actions">{actions}</div> : null}
    </section>
  );
}
