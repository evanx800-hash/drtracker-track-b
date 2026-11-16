import sys
from streamlit.testing.v1 import AppTest
pages = ["views/overview.py","views/parse.py","views/review.py","views/queries.py","views/evaluation.py","views/schema.py"]
for pg in pages:
    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60)
    at.run()
    at.switch_page(pg).run()
    print(pg, "EXC:" , [e.value for e in at.exception] if at.exception else "none")
# parse flow
at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60); at.run(); at.switch_page("views/parse.py").run()
at.button[0].click().run()
print("parse exc", [e.value for e in at.exception])
print([d.value.shape for d in at.dataframe])
