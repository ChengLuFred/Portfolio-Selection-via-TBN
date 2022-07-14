import pandas as pd

def export_dataframe_to_latex_table(
                                    df: pd.DataFrame, 
                                    table_name: str,
                                    output_path: str = '/Users/cheng/Dropbox/Apps/Overleaf/Weekly Report Cheng/table',
                                    float_format = "%.3f",
                                    caption: str = '',
                                    label:str = ''
                                    ) -> str:
                                    
            output_file_path = output_path + '/' + table_name + '.tex'
            caption = r'\textbf{' + table_name + r'} \\ ' + caption
            
            latex_table = df.to_latex(output_file_path, 
                                    float_format=float_format, 
                                    caption=caption, 
                                    label=label,
                                    position = 'h!')

            print('write table to {}'.format(output_file_path))

            return latex_table