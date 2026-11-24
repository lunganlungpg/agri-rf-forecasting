# Container image published to GitHub Packages as
#   ghcr.io/lunganlungpg/agri-rf-forecasting
#
# It bundles BOTH implementations so nothing has to be installed locally:
#   docker run --rm ghcr.io/lunganlungpg/agri-rf-forecasting                 # Python CV pipeline
#   docker run --rm ghcr.io/lunganlungpg/agri-rf-forecasting Rscript -e 'agrirf::rf_run_pipeline()'
#   docker run --rm -v "$PWD/out:/opt/agri-rf/plots" ghcr.io/lunganlungpg/agri-rf-forecasting agri-rf plots

# rocker/r-ver = Ubuntu + pinned R, preconfigured for fast binary CRAN installs.
FROM rocker/r-ver:4.4.2

LABEL org.opencontainers.image.title="agri-rf-forecasting" \
      org.opencontainers.image.description="Random Forest agricultural yield forecasting (Python + R)" \
      org.opencontainers.image.licenses="MIT"

# Python runtime (venv is used because Ubuntu blocks system-wide pip installs).
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/agri-rf
COPY . .

# 1) Python package `agrirf` + CLI `agri-rf`
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir .
ENV PATH="/opt/venv/bin:${PATH}"

# 2) R package `agrirf` and its dependency randomForest
RUN R -q -e 'install.packages("randomForest")' \
 && R CMD INSTALL .

# Fail the build early if either package cannot be loaded.
RUN python -c "import agrirf; print('agrirf (Python)', agrirf.__version__)" \
 && Rscript -e 'library(agrirf); cat("agrirf (R)", as.character(packageVersion("agrirf")), "\n")'

CMD ["agri-rf", "forecast"]
