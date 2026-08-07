from __future__ import annotations
from .probability import ProbabilityResult


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x*100:.0f}%"


def plain_summary(intra: ProbabilityResult | None, daily: list[ProbabilityResult], gamma=None) -> str:
    d3=next((x for x in daily if x.horizon=="3D"),None)
    d5=next((x for x in daily if x.horizon=="5D"),None)
    parts=[]
    if intra:
        up=intra.probabilities.get("close_up",0.5)
        if up>=0.60: parts.append("Para lo que queda de hoy, el histórico parecido favorece subida.")
        elif up<=0.40: parts.append("Para lo que queda de hoy, el histórico parecido favorece debilidad.")
        else: parts.append("Para hoy no hay una ventaja direccional clara.")
    if d3:
        p5=d3.probabilities.get("touch_+5%")
        m5=d3.probabilities.get("touch_-5%")
        if p5 is not None and m5 is not None:
            if p5-m5>0.15: parts.append("A 3 días, el balance de recorridos es favorable al alza.")
            elif m5-p5>0.15: parts.append("A 3 días, aumenta el riesgo de corrección.")
            else: parts.append("A 3 días, el reparto entre subida y caída es bastante equilibrado.")
    if gamma:
        if gamma.call_wall: parts.append(f"La Call Wall estimada está en ${gamma.call_wall:.2f}.")
        if gamma.put_wall: parts.append(f"La Put Wall estimada está en ${gamma.put_wall:.2f}.")
        if gamma.gamma_flip_proxy: parts.append(f"El Gamma Flip proxy está cerca de ${gamma.gamma_flip_proxy:.2f}.")
    if d5 and d5.confidence=="BAJA":
        parts.append("La lectura a 5 días tiene baja confianza: no debe usarse sola para operar.")
    return " ".join(parts)
