FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        latexmk \
        texlive-fonts-recommended \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-latex-recommended \
        texlive-lang-japanese \
        texlive-luatex \
        texlive-pictures \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/paper
CMD ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"]
