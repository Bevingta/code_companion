import requests
from bs4 import BeautifulSoup

def get_cvss_info(cve_id):
    url = f"https://nvd.nist.gov/vuln/detail/{cve_id}"
    response = requests.get(url)
  
    if response.status_code == 200:
       soup = BeautifulSoup(response.text, 'html.parser')
      
       # Find the CVSS calculator anchor with id "Cvss2CalculatorAnchor"
       cvss_anchor = soup.find(id="Cvss2CalculatorAnchor")
      
       if cvss_anchor:
           # Extract the severity score and level
           severity_text = cvss_anchor.text.strip()  # "5.0 MEDIUM"
          
           # Extract the CVSS vector from the href attribute
           href = cvss_anchor.get('href', '')
           vector_start = href.find('vector=(')
           vector_end = href.find(')&', vector_start)
          
           if vector_start != -1 and vector_end != -1:
               vector = href[vector_start+8:vector_end]
              
               return {
                   'severity_text': severity_text,
                   'score': severity_text.split()[0],
                   'level': severity_text.split()[1]
               }
    return None