import type { ReactNode } from "react";

interface PageHeaderProps {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}

export default function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: PageHeaderProps) {
  return (
    <div className="flex flex-col gap-4 border-b border-border pb-6 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0 space-y-3">
        <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
          <span className="size-1.5 shrink-0 bg-primary" aria-hidden="true" />
          <span>{eyebrow}</span>
        </div>

        <div className="space-y-2">
          <h2 className="font-heading text-2xl font-semibold tracking-tight md:text-3xl">
            {title}
          </h2>
          <p className="max-w-2xl text-xs leading-6 text-muted-foreground">
            {description}
          </p>
        </div>
      </div>

      {actions ? <div className="shrink-0">{actions}</div> : null}
    </div>
  );
}
