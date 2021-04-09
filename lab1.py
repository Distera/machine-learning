from pandas_profiling import ProfileReport


def lab1(data):
    ProfileReport(data, title='Pandas Profiling Report', explorative=True).to_file("report.html")
