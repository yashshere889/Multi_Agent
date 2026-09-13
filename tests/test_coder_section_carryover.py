"""A targeted fix must not drop what the sections it reused still use.

Barkla job 10496130 spent three attempts ping-ponging: regenerating load_data
dropped the PCA import run_experiment needed, and regenerating run_experiment
dropped the requests/StringIO imports and DATA_URL constant load_data needed.
"""

from research_pipeline.agents.coder import repair

PREVIOUS_IMPORTS = (
    "import numpy as np\n"
    "import requests\n"
    "from io import StringIO\n"
    "from sklearn.decomposition import PCA\n"
)


def test_a_dropped_import_the_reused_code_needs_is_put_back():
    regenerated = "import numpy as np\nimport requests\nfrom io import StringIO\n"

    section, carried = repair.carry_over_definitions(PREVIOUS_IMPORTS, regenerated, {"PCA", "np"})

    assert carried == ["PCA"]
    assert "from sklearn.decomposition import PCA" in section
    assert section.count("import numpy as np") == 1


def test_a_dropped_import_nothing_reused_needs_stays_dropped():
    """A fix that removed an import on purpose (a name that does not exist) keeps it removed."""
    previous = "from SALib.sample import sample_saltelli\nimport numpy as np\n"
    regenerated = "from SALib.sample import saltelli\nimport numpy as np\n"

    section, carried = repair.carry_over_definitions(previous, regenerated, {"np", "saltelli"})

    assert carried == []
    assert section == regenerated


def test_a_dropped_constant_comes_back_without_overriding_the_new_ones():
    previous = "DATA_URL = 'https://example.org/data.csv'\nSEED = 1\n"
    regenerated = "SEED = 2\n"

    section, carried = repair.carry_over_definitions(previous, regenerated, {"DATA_URL", "SEED"})

    assert carried == ["DATA_URL"]
    assert "DATA_URL = 'https://example.org/data.csv'" in section
    assert "SEED = 2" in section and "SEED = 1" not in section


def test_a_dropped_helper_comes_back_with_its_decorator():
    previous = "import functools\n\n@functools.lru_cache\ndef variance_decomposition(x):\n    return x\n"
    regenerated = "import functools\n"

    section, carried = repair.carry_over_definitions(previous, regenerated, {"variance_decomposition"})

    assert carried == ["variance_decomposition"]
    assert "@functools.lru_cache\ndef variance_decomposition(x):" in section


def test_an_unparseable_side_leaves_the_regeneration_as_it_is():
    assert repair.carry_over_definitions("import (", "import numpy as np\n", {"np"}) == ("import numpy as np\n", [])
    assert repair.carry_over_definitions(PREVIOUS_IMPORTS, "def broken(:\n", {"PCA"}) == ("def broken(:\n", [])


def test_referenced_names_reads_bare_names_and_attribute_bases():
    source = "def load_data():\n    response = requests.get(DATA_URL)\n    return pd.read_csv(StringIO(response.text))\n"
    assert {"requests", "DATA_URL", "pd", "StringIO", "response"} <= repair.referenced_names(source)
