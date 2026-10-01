param(
    [Parameter(Mandatory = $true)][string]$DocxPath,
    [Parameter(Mandatory = $true)][string]$PdfPath,
    [Parameter(Mandatory = $true)][string]$ResultJson
)

$ErrorActionPreference = 'Stop'
$word = $null
$doc = $null
$simSunFont = [string]::Concat([char]0x5B8B, [char]0x4F53)

function Set-ExplicitTitleFont {
    param([Parameter(Mandatory = $true)]$Range)

    $Range.Font.Name = 'Times New Roman'
    $Range.Font.NameAscii = 'Times New Roman'
    $Range.Font.NameOther = 'Times New Roman'
    $Range.Font.NameFarEast = $simSunFont
}

try {
    $docx = [System.IO.Path]::GetFullPath($DocxPath)
    $pdf = [System.IO.Path]::GetFullPath($PdfPath)
    $result = [System.IO.Path]::GetFullPath($ResultJson)

    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($pdf)) | Out-Null
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($result)) | Out-Null

    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.Options.UpdateFieldsAtPrint = $true
    $word.Options.UpdateLinksAtPrint = $true

    $doc = $word.Documents.Open($docx, $false, $false)

    foreach ($toc in $doc.TablesOfContents) {
        $toc.Update()
        $toc.UpdatePageNumbers()
        Set-ExplicitTitleFont -Range $toc.Range
    }

    foreach ($storyType in 1..17) {
        try {
            $story = $doc.StoryRanges.Item($storyType)
            while ($null -ne $story) {
                if ($story.Fields.Count -gt 0) {
                    $story.Fields.Update() | Out-Null
                }
                $story = $story.NextStoryRange
            }
        } catch {
            # Not every story type exists in every document.
        }
    }

    if ($doc.Fields.Count -gt 0) {
        $doc.Fields.Update() | Out-Null
    }
    foreach ($toc in $doc.TablesOfContents) {
        $toc.Update()
        $toc.UpdatePageNumbers()
        Set-ExplicitTitleFont -Range $toc.Range
    }

    foreach ($paragraph in $doc.Paragraphs) {
        $styleName = [string]$paragraph.Style.NameLocal
        $outlineLevel = [int]$paragraph.OutlineLevel
        $isHeading = ($outlineLevel -ge 1 -and $outlineLevel -le 3)
        $isCaption = $styleName -eq 'Caption'
        $alignment = [int]$paragraph.Range.ParagraphFormat.Alignment
        $fontSize = [single]$paragraph.Range.Font.Size
        $isFrontTitle = ($alignment -eq 1 -and $fontSize -ge 15 -and $fontSize -lt 999999)
        if ($isHeading -or $isCaption -or $isFrontTitle) {
            Set-ExplicitTitleFont -Range $paragraph.Range
        }
    }

    $doc.Repaginate()
    $doc.Save()
    $pageCount = $doc.ComputeStatistics(2)
    $wordCount = $doc.ComputeStatistics(0)
    $characterCount = $doc.ComputeStatistics(3)
    $wordVersion = $word.Version

    if (Test-Path -LiteralPath $pdf) {
        Remove-Item -LiteralPath $pdf -Force
    }
    $doc.ExportAsFixedFormat($pdf, 17)

    [ordered]@{
        export_engine = 'Microsoft Word'
        word_version = [string]$wordVersion
        page_count = [int]$pageCount
        word_count = [int]$wordCount
        character_count = [int]$characterCount
        docx = $docx
        pdf = $pdf
        pdf_exists = (Test-Path -LiteralPath $pdf)
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $result -Encoding UTF8

    $doc.Close(0)
    $doc = $null
    $word.Quit()
    $word = $null
} finally {
    if ($null -ne $doc) {
        try { $doc.Close(0) } catch {}
    }
    if ($null -ne $word) {
        try { $word.Quit() } catch {}
    }
    if ($null -ne $doc) {
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($doc)
    }
    if ($null -ne $word) {
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
