import type { LucideIcon } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface MetricCardProps {
  title: string;
  symbol: string;
  description: string;
  formula: string;
  icon: LucideIcon;
}

export default function MetricCard({
  title,
  symbol,
  description,
  formula,
  icon: Icon,
}: MetricCardProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Icon className="size-4 text-muted-foreground" />
            <CardTitle>{title}</CardTitle>
          </div>
          <span className="font-heading text-sm text-muted-foreground">
            {symbol}
          </span>
        </div>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="border border-border bg-muted/20 px-3 py-2">
          <p className="font-heading text-[11px]">{formula}</p>
        </div>
      </CardContent>
    </Card>
  );
}
