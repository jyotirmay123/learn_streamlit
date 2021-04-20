FROM python:3.8

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8080

COPY . /app

CMD find /usr/local/lib/python3.8/site-packages/streamlit -type f \( -iname \*.py -o -iname \*.js \) -print0 | xargs -0 sed -i 's/healthz/health-check/g' && streamlit run compare_forecast.py --server.port 8080 --server.enableCORS=false --server.enableXsrfProtection=false