import typing
import os

from jinja2 import Environment, loaders


def render_sql_from_file(
    path: str, additional_searchpaths: typing.Tuple[str, ...] = (), **kwargs
) -> str:
    """Using jinja2 to render given template at `path`.

    You can provide additional filesystem `searchpaths`. By default searches for
    files in `common/dataset/queries/`. Additional search paths might override
    default query templates in the common package.

    See also:
      - :class:`jinja2.FileSystemLoader`
      - :meth:`jinja2.Template.render`
    """
    default_searchpaths = (
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queries'),
    )
    env = Environment(
        loader=loaders.FileSystemLoader(additional_searchpaths + default_searchpaths),
    )
    return env.get_template(path).render(**kwargs)
