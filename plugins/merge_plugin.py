import os
from pelican import signals
import logging


logger = logging.getLogger(__name__)


def merge_md_with_header(pelican):
    """
    Merge .md.header file content to the top of the corresponding .md file.
    """

    logger.info("Merging .md files with headers...")
    content_path = os.path.join(pelican.settings["PATH"], "content")

    for root, _, files in os.walk(content_path):
        for file in files:
            if not file.endswith(".md"):
                continue

            md_file_path = os.path.join(root, file)
            header_file_path = os.path.join(root, file + ".header")
            merge_file_path = os.path.join(root, "merge_" + file)
            content_file_path = md_file_path + ".content"

            if not os.path.exists(header_file_path):
                raise FileNotFoundError(f"Header file '{header_file_path}' not found.")

            with open(header_file_path, "r", encoding="utf-8") as header_file:
                header_content = header_file.read()

            with open(md_file_path, "r", encoding="utf-8") as md_file:
                md_content = md_file.read()
            os.rename(md_file_path, content_file_path)

            # Combine header and markdown content
            combined_content = header_content + "\n\n" + md_content

            # Write the combined content back to the markdown file
            with open(merge_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(combined_content)

    logger.info("Finished merging .md files with headers.")


def end_merge(pelican):
    """
    End Merge delete startwith merge_ and endwith .md files, rename endwith .content files
    """
    logger.info("Ending merge and restoring .md files...")
    content_path = os.path.join(pelican.settings["PATH"], "content")

    for root, _, files in os.walk(content_path):
        for file in files:
            if file.startswith("merge_") and file.endswith(".md"):
                os.remove(os.path.join(root, file))

            elif file.endswith(".md.content"):
                content_path = os.path.join(root, file)
                rename_path = content_path[:-8]
                os.rename(content_path, rename_path)


def register():
    """
    Register the merge_md_with_header and end_merge functions with the Pelican signals.
    """
    signals.initialized.connect(merge_md_with_header)
    signals.finalized.connect(end_merge)
